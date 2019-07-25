import numpy as np
import tensorflow as tf
import time
import keras.datasets as datasets

def forward_prop(X, device):

	use_device = '/cpu:0' 
	if(device == 'gpu'):
		use_device = '/gpu:0'

	with tf.device(use_device):

	    Z1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same', )
	    A1 = tf.nn.relu(Z1)
	    P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	    #comentario
	    Z2 = tf.layers.conv2d(inputs = P1, filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')
	    A2 = tf.nn.relu(Z2)
	    P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	    #comentario
	    P2 = tf.contrib.layers.flatten(P2) #tf.layer.dense
	    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)

	return Z3


def compute_cost(log_probs, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=log_probs, labels=Y))
    
    return cost


#=============================================================================================
(x_train, y_train),(x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train[0:50000]
y_train = y_train[0:50000]


x_train = x_train / 255
x_test = x_test / 255
m, n_H0, n_W0, n_C0 = x_train.shape
num_classes = 10
m_test = x_test.shape[0]
device = 'cpu'

categ_y_train = np.zeros((m, num_classes))
categ_y_train[np.arange(m), np.squeeze(y_train)] = 1

minibatch_size = 10
num_minibatches = int(m / minibatch_size)
learning_rate = 0.001
num_epochs = 50
costs = []

dataset = tf.data.Dataset.from_tensor_slices((x_train, categ_y_train)).batch(minibatch_size)
iterator = dataset.make_initializable_iterator()
batch_x_train, batch_y_train = iterator.get_next() 

init = tf.global_variables_initializer()
iter_init = iterator.initializer

log_probs = forward_prop(batch_x_train, device)
#probs = tf.nn.softmax(log_probs)
#predict_class = tf.math.argmax(log_probs, 1)
#correct_prediction = tf.math.equal(predict_class, tf.math.argmax(batch_y_train, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#tf_acc = tf.metrics.accuracy(labels = tf.math.argmax(Y, 1), predictions = predict_class)
cost = compute_cost(log_probs, batch_y_train)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()


start = time.time()
with tf.Session() as sess:

        sess.run(init)
        
        for epoch in range(num_epochs + 1):

            minibatch_cost = 0.
            minibatch_acc = 0.
            sess.run(iter_init)

            for i in range(num_minibatches - 1):
            	_, temp_cost = sess.run([optimizer, cost])
            	#_, temp_cost = sess.run([optimizer, cost])
            	minibatch_cost += temp_cost / num_minibatches
            	#minibatch_acc += temp_acc / num_minibatches

               
            if epoch % 5 == 0:
                print ('Cost after epoch {:d}:'.format(epoch), '{0:.4f}'.format(minibatch_cost))
                #train_accuracy = sess.run(accuracy, feed_dict = {X: x_train, Y: y_train}) #Si esto no se comenta peta por memoria
                #test_accuracy = sess.run(accuracy, feed_dict = {X: x_test[0:1000], Y: y_test[0:1000]})
                #tf_acc = sess.run(tf_acc, feed_dict = {X: x_test[0:1000], Y: y_test[0:1000]})
                #print("Train Accuracy:", train_accuracy)
                #print("Test Accuracy:", test_accuracy) 
                #print('tf_acc:', tf_acc)
                
            #if epoch % 1 == 0:
                #costs.append(minibatch_cost)
			

end = time.time()
print('Time using', device, '{0:.2f}'.format((end - start) / 60), 'mins')

