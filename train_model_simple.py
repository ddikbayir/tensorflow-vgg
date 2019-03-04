import json
import sys
import os
import os.path as pth
import vgg
import tensorflow as tf
import numpy as np
import time
import layers as L
import tools
import dataset
import yaml
import argparse
import time
from tensorflow.python.client import timeline
# =====================================
# Training configuration default params
# =====================================
config = {}

#################################################################

# customize your model here
# =========================
def build_model(input_data_tensor, input_label_tensor):
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]
    images = tf.image.resize_images(input_data_tensor, [224, 224])
    logits = vgg.build(images, n_classes=num_classes, training=True)
    probs = tf.nn.softmax(logits)
    loss_classify = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))
    loss = loss_classify + weight_decay*loss_weight_decay
    error_top5 = L.topK_error(probs, input_label_tensor, K=5)
    error_top1 = L.topK_error(probs, input_label_tensor, K=1)

    # you must return a dictionary with loss as a key, other variables
    return dict(loss=loss,
                probs=probs,
                logits=logits,
                error_top5=error_top5,
                error_top1=error_top1)


def train(trn_data_generator, vld_data=None):
    learning_rate = config['learning_rate']
    experiment_dir = config['experiment_dir']
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config["num_samples_per_epoch"]
    steps_per_epoch = num_samples_per_epoch // batch_size
    num_steps = steps_per_epoch * num_epochs
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    train_log_fpath = pth.join(experiment_dir, 'train.log')
    vld_iter = config["vld_iter"]
    checkpoint_iter = config["checkpoint_iter"]
    pretrained_weights = config.get("pretrained_weights", None)

    # ========================
    # construct training graph
    # ========================
    G = tf.Graph()
    with G.as_default():
        input_data_tensor = tf.placeholder(tf.float32, [None] + data_dims)
        input_label_tensor = tf.placeholder(tf.int32, [None])
        model = build_model(input_data_tensor, input_label_tensor)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(model["loss"])
        grad_step = optimizer.apply_gradients(grads)
        init = tf.initialize_all_variables()


    # ===================================
    # initialize and run training session
    # ===================================
    
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=G, config=config_proto)

    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)


    def profile(run_metadata, epoch=0):
        with open('profs/timeline_step' + str(epoch) + '.json', 'w') as f:
            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            f.write(chrome_trace)

    sess.run(init, run_metadata=run_metadata, options=options)
    profile(run_metadata, -1)
    
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        if pretrained_weights:
            print("-- loading weights from %s" % pretrained_weights)
            tools.load_weights(G, pretrained_weights)


        # Start training loop
        for step in range(num_steps):
            batch_train = trn_data_generator.next()
            X_trn = np.array(batch_train[0])
            Y_trn = np.array(batch_train[1])

            ops = [grad_step] + [model[k] for k in sorted(model.keys())]
            inputs = {input_data_tensor: X_trn, input_label_tensor: Y_trn}
            start_time = time.time()
	    results = sess.run(ops, feed_dict=inputs, run_metadata=run_metadata, options=options)
            elapsed = time.time() - start_time
	    
	    samples_p_second = 0
	    print("Ex/sec: %.1f" % batch_size/float(elapsed))
	    profile(run_metadata, step)

            results = dict(zip(sorted(model.keys()), results[1:]))
            print("TRN step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (step,
                                                                                 results["error_top1"],
                                                                                 results["error_top5"],
                                                                                 results["loss"]))
            

            # report evaluation metrics every 10 training steps
            if (step % vld_iter == 0):
                print("-- running evaluation on vld split")
                X_vld = vld_data[0]
                Y_vld = vld_data[1]
                inputs = [input_data_tensor, input_label_tensor]
                args = [X_vld, Y_vld]
                ops = [model[k] for k in sorted(model.keys())]
                results = tools.iterative_reduce(ops, inputs, args, batch_size=1, fn=lambda x: np.mean(x, axis=0))
                results = dict(zip(sorted(model.keys()), results))
                print("VLD step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (step,
                                                                                     results["error_top1"],
                                                                                     results["error_top5"],
                                                                                     results["loss"]))
             

            if (step % checkpoint_iter == 0) or (step + 1 == num_steps):
                print("-- saving check point")
                tools.save_weights(G, pth.join(checkpoint_dir, "weights.%s" % step))
    options = tf.profiler.ProfileOptionBuilder.time_and_memory()
    options["min_bytes"] = 0
    options["min_micros"] = 0
    options["output"] = 'file:outfile=ooo.txt'
    options["select"] = ("bytes", "peak_bytes", "output_bytes",
                            "residual_bytes")
    mem = tf.profiler.profile(tf.Graph(), run_meta=run_metadata, cmd="scope", options=options)
    with open('profs/mem.txt', 'w') as f:
        f.write(str(mem))


    operations_tensors = {}
    operations_names = tf.get_default_graph().get_operations()
    count1 = 0
    count2 = 0

    for operation in operations_names:
        operation_name = operation.name
        operations_info = tf.get_default_graph().get_operation_by_name(operation_name).values()
        if len(operations_info) > 0:
            if not (operations_info[0].shape.ndims is None):
                operation_shape = operations_info[0].shape.as_list()
                operation_dtype_size = operations_info[0].dtype.size
                if not (operation_dtype_size is None):
                    operation_no_of_elements = 1
                    for dim in operation_shape:
                        if not(dim is None):
                            operation_no_of_elements = operation_no_of_elements * dim
                    total_size = operation_no_of_elements * operation_dtype_size
                    operations_tensors[operation_name] = total_size
                else:
                    count1 = count1 + 1
            else:
                count1 = count1 + 1
                operations_tensors[operation_name] = -1

            #   print('no shape_1: ' + operation_name)
            #  print('no shape_2: ' + str(operations_info))
            #  operation_namee = operation_name + ':0'
            # tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
            # print('no shape_3:' + str(tf.shape(tensor)))
            # print('no shape:' + str(tensor.get_shape()))

        else:
            # print('no info :' + operation_name)
            # operation_namee = operation.name + ':0'
            count2 = count2 + 1
            operations_tensors[operation_name] = -1

            # try:
            #   tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
            # print(tensor)
            # print(tf.shape(tensor))
            # except:
            # print('no tensor: ' + operation_namee)
    print(count1)
    print(count2)

    with open('tensors_sz.json', 'w') as f:
        json.dump(operations_tensors, f)

def main():
    batch_size = config['batch_size']
    experiment_dir = config['experiment_dir']

    # setup experiment and checkpoint directories
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    trn_data_generator, vld_data = dataset.get_cifar10(batch_size)
    train(trn_data_generator, vld_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

        print "Experiment config"
        print "------------------"
        print json.dumps(config, indent=4)
        print "------------------"
    main()
