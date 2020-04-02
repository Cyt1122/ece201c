import tensorflow as tf
import numpy as np

def get_prune_op(sess, train_gradient, percentage=0.5):
    all_variables = sess.graph.get_collection('variables')
    # ================================================================ #
    # YOUR CODE HERE:
    #   1.find prunable variables i.e. kernel weight/bias
    #   2.prune parameters based on your threshold, specified by input argument percentage
    #   3.get pruned gradient update operator accordingly, save to prune_gradient
    #   example pseudo code for step 2-3:
    #       for var in prunable_variables:
    #           var_np = var.eval()
    #           # construct pruning mask
    #           new_var_np = mask * var_np
    #           prune_op = var.assign(new_var_np)
    #           # apply parameter pruning by sess.run(prune_op)
    #           prune_gradient = train_gradient * mask for each parameter
    #               Example:
    #                   mask = np.zeros(96, dtype=np.float32)
    #                   mask_tensor = tf.convert_to_tensor(mask)
    #                   prune_gradient = [[tf.multiply(v[0], mask_tensor), v[1]] for v in train_gradient if 'fcn/conv1-1/bias' in v[1].name]
    #        return prune_gradient   
    # ================================================================ #
    # modify here, no change on gradient for now
    i = 0
    prune_gradient = []
    #print(all_variables)
    #print('----------')
    #print('----------')
    #print(all_variables[5].name)
    #print(train_gradient[2][1].name)
    var_np_pr = 0
    for var in all_variables:
        var_np = var.eval(session=sess)
        var_np_flat = np.sort(np.abs(np.reshape(var_np, [1, -1])))
        var_np_flat = var_np_flat[np.nonzero(var_np_flat)]
        indx = np.size(var_np_flat) * percentage
        threshold = var_np_flat[int(indx - 1)]
        mask = np.where(np.abs(var_np) <= threshold, 0, 1)
        mask_tf = tf.convert_to_tensor(mask)
        var_np_pr = mask * var_np
        prune_op = var.assign(var_np_pr)
        sess.run(prune_op)
        if var.name == train_gradient[i][1].name:
            temp = tf.multiply(train_gradient[i][0], tf.cast(mask_tf,tf.float32))
            tuplep = (temp, train_gradient[i][1])
            prune_gradient.append(tuplep)
            i = i + 1
    '''
    print('----------')
    print('----------')
    print('----------')
    print('----------')
    print('after mul')
    for a in train_gradient:
        print(a)
        print('----------')
    print('----------')
    print('----------')
    print('before mul')
    for a in prune_gradient:
        print(a)
        print('----------')
    print('----------')
    print('----------')
    print('mask')
    for a in msk:
        print(a)
        print('----------')
    print('----------')
    print('----------')
    print('----------')
    print('----------')
    '''
    #prune_gradient = train_gradient
    #train_gradient = train_gradient.eval(session=sess)
    #print(train_gradient[1].name)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return prune_gradient