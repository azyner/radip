import tensorflow as tf
import numpy as np

# Library that implements Alex Graves 2014 paper

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - 2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)),
                                                                                 s1s2)
    negRho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * negRho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
    result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    # implementing eq # 26 of http://arxiv.org/abs/1308.0850
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    result = -tf.log(tf.maximum(result1, 1e-20))  # at the beginning, some errors are exactly zero.

    return result
    #return tf.reduce_sum(result)


def lossfunc_wrapper(labels, logits):
    # Because the library cannot believe seq2seq without logits is a thing.
    ground_truth = labels
    prediction = logits
    # TODO only compare first two digits
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = get_mixture_coef(prediction)
    #HACK to force NaN's so I can write a catcher
    #z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(1,6,prediction)
    x1_data, x2_data, heading, speed = tf.split(axis=1,num_or_size_splits=4,value=ground_truth)
    return get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data)


# below is where we need to do MDN splitting of distribution params
# Temperature param should only be used during sampling. The other functions record the mixtures for visualisation.
def get_mixture_coef(output, temperature=None):
    # returns the tf slices containing mdn dist params
    # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
    z = output
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(axis=1, num_or_size_splits=6, value=z)

    # process output z's into MDN paramters
    # softmax all the pi's:
    # max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
    # z_pi = tf.subtract(z_pi, max_pi)
    # z_pi = tf.exp(z_pi)
    # normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))
    # z_pi = tf.multiply(normalize_pi, z_pi)
    if temperature==None:
        z_pi = tf.nn.softmax(z_pi)
    else:
        z_pi = tf.nn.softmax(tf.divide(z_pi, temperature))

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    # Bound the correlation coefficient to within 1,-1
    z_corr = tf.minimum(0.999,tf.maximum(-0.999,tf.tanh(z_corr)))

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]


def sample(output, temperature=1.0):
    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = get_mixture_coef(output, temperature=temperature)
    # Take in output params
    # return a single sample used for sequence generation / loop-back

    # I have to replace these functions with tf ones.
    # Replace this with tf.multinomial
    def get_pi_idx(x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print 'error with sampling ensemble'
        return -1

    #There is a random_normal
    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0):
        # mean = [mu1, mu2]
        #cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        # input temp = 1.0
        # s1 *= temp * temp # same for s2

        # During checkpoint loading for best params, it becomes float64 for some reason
        s1 = tf.multiply(tf.square(tf.to_float(temp)), s1)
        s2 = tf.multiply(tf.square(tf.to_float(temp)), s2)

        covUL = tf.expand_dims(tf.square(s1), 1)
        covUR = tf.expand_dims(tf.multiply(rho, tf.multiply(s1, s2)), 1)
        covLL = tf.expand_dims(tf.multiply(rho, tf.multiply(s1, s2)), 1)
        covLR = tf.expand_dims(tf.square(s2), 1)

        covU = tf.expand_dims(tf.concat(axis=1, values=[covUL, covUR]), 2)
        covL = tf.expand_dims(tf.concat(axis=1, values=[covLL, covLR]), 2)
        cov = tf.concat(axis=2, values=[covU, covL])

        # #tf.random_normal? its not multivariate, but it will have to do.
        # #tf.self_adjoint_eigvals can be used on the cov matrix
        #
        # x = np.random.multivariate_normal(mean, cov, 1)
        # return x[0][0], x[0][1]

        #See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        #step 2

        batch_size = tf.shape(mu1)
        #batch_size = mu1.get_shape()
        convar = tf.constant([2])
        random_shape = tf.concat(axis=0, values=[convar, batch_size])

        z = tf.expand_dims(tf.transpose(tf.random_normal(random_shape)), 2)

        L = tf.cholesky(cov)
        mean = tf.concat(axis=1, values=[tf.expand_dims(mu1, 1),
                         tf.expand_dims(mu2, 1)])
        Lz = tf.squeeze(tf.matmul(L, z), [2])
        x = tf.add(mean, Lz)

        return x

    # Now pick one of the N mixtures using the pi prob dist.
    # tf multinomial wants the `unnormalized log probabilities', which explains the extra tf.log
    idx = tf.to_int32(tf.multinomial(tf.log(o_pi), 1))

    #TODO - gather_nd does not have a gradient function. Replace with:
    # 1 - convert batch_idx to 1 hot vector
    # 2 - multiply
    # 3 - reduce-sum
    #Because the documentation for gather_nd is easier to read than tf.gather
    batch_range = tf.expand_dims(tf.range(0,idx.get_shape()[0]),1) # make the first idx for batch_idx a self refencing idx
    batch_idx = tf.concat(values=[batch_range,idx],axis=1)         # then add the MDN idx.
    next = sample_gaussian_2d(tf.gather_nd(o_mu1,batch_idx),
                              tf.gather_nd(o_mu2,batch_idx),
                              tf.gather_nd(o_sigma1,batch_idx),
                              tf.gather_nd(o_sigma2,batch_idx),
                              tf.gather_nd(o_corr,batch_idx), temp=temperature)

    return next


def compute_derivates(output_prev, output_current, network_input_columns,
                      velocity_threshold=tf.constant(2.0, dtype=tf.float32), subsample_rate=1):
    # ['easting', 'northing', 'heading', 'speed']
    # Assume the first two are x and y
    if 'heading' not in network_input_columns[2] or \
       network_input_columns[3] is not 'speed':
        print "not implemented yet"
        exit()

    # column 2 is heading, so do some trig,
    # column 3 is speed, so its just a subtraction and vector magnitude
    x_p, y_p, heading_p, speed_p = tf.split(output_prev, 4, axis=1)
    x_c, y_c = tf.split(output_current, 2, axis=1)
    pos_d_i = tf.complex(tf.subtract(x_p,x_c), tf.subtract(y_p, y_c))  # Define x,y as a complex number
    pos_d = tf.abs(pos_d_i)  # Use abs to get magnitude
    print "Warning, velocity loop-back generator assumes data was recorded at 25 Hz"
    v_c = tf.multiply(pos_d, (25/subsample_rate))  # delta * Hz = number of meters per second
    #  For whatever reason, atan2 convention is atan2(y,x)
    h_c = tf.atan2(tf.subtract(y_c, y_p), tf.subtract(x_c, x_p))
    # TODO Element wise, I have to condition on speed. If < 2m/s (hyper-parameter?) use old heading, else compute heading
    # I don't want to use tf.cond as it does not perform element-wise logic.
    # So I'm going to construct this fundamentally - Multiply by zero or one and sum
    use_old_heading = tf.less(v_c, velocity_threshold) # Broadcasting will up-size the scalar to a vector
    use_new_heading = tf.logical_not(use_old_heading)
    use_old_heading, use_new_heading = (tf.to_float(use_old_heading), tf.to_float(use_new_heading))
    new_heading = tf.add(tf.multiply(use_old_heading, heading_p), tf.multiply(use_new_heading, h_c))
    output_with_extras = tf.concat([x_c, y_c, new_heading, v_c], axis=1)

    return output_with_extras


def upscale_and_resolve_mixtures(output, scaling_layer):
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = get_mixture_coef(output)
    z_mu1 = tf.add(tf.multiply(z_mu1, scaling_layer[1][0]), scaling_layer[0][0])
    z_mu2 = tf.add(tf.multiply(z_mu2, scaling_layer[1][1]), scaling_layer[0][1])

    z_sigma1 = tf.multiply(z_sigma1, scaling_layer[1][0])
    z_sigma2 = tf.multiply(z_sigma2, scaling_layer[1][1])

    return tf.concat([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr],axis=1)