from loading_blocks import *
import os
import tensorflow as tf


NB_L1 = 100
NB_L2 = 50

def create_placeholders(nb_features,nb_outputs):
	" Create placeholder for X and Y "
	X = tf.placeholder(tf.float32, shape=(nb_features, None), name="X")
	Y = tf.placeholder(tf.float32, shape=(nb_outputs, None), name="Y")
	return X,Y

def initialize_parameters(nb_features,nb_l1,nb_l2,nb_outputs) :
	W1 = tf.get_variable("W1", [nb_l1,nb_features], initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [nb_l1,1], initializer=tf.zeros_initializer())
	W2 = tf.get_variable("W2", [nb_l2,nb_l1], initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [nb_l2,1], initializer=tf.zeros_initializer())
	W3 = tf.get_variable("W3", [nb_outputs,nb_l2], initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [nb_outputs,1], initializer=tf.zeros_initializer())

	parameters = {
		"W1" : W1,
		"b1" : b1,
		"W2" : W2,
		"b2" : b2,
		"W3" : W3,
		"b3" : b3
	}
	return parameters

def forward_propagation(X,parameters) :
	"forward passage of the Neural network RELU-->RELU-->LINEAR"

	W1= parameters['W1']
	b1 = parameters['b1']
	W2= parameters['W2']
	b2 = parameters['b2']
	W3= parameters['W3']
	b3 = parameters['b3']



	Z1 = tf.add(tf.matmul(W1,X),b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)

	return Z3

def compute_cost(Z3,Y) :
	"Compute the cost in accordance to the results using MSE "
	cost = tf.reduce_mean(tf.squared_difference(Z3,Y))
	return cost


def write_submission(Y_submit,X_submit,file_name='submission.csv'):
	y_submit=pd.DataFrame()
	y_submit['Id']= [i for i in range(1461,1461+X_submit.shape[1])]
	y_submit['SalePrice']= Y_submit.T
	y_submit.to_csv(file_name,index=False)


def model(X_train, Y_train, X_test, Y_test,X_Submit,max_epoch=100000, learning_rate=0.03):
	tf.reset_default_graph()
	(nb_features,m) = X_train.shape
	nb_outputs = Y_train.shape[0]

	X,Y = create_placeholders(nb_features,nb_outputs)
	parameters = initialize_parameters(nb_features,NB_L1,NB_L2,nb_outputs)
	Z3 = forward_propagation(X,parameters)
	cost = compute_cost(Z3,Y)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	##### SUMMARIES

	train_cost = tf.placeholder(tf.float32, shape=None, name='train_cost')
	train_cost_summary = tf.summary.scalar('train cost ', train_cost)

	test_cost = tf.placeholder(tf.float32, shape=None, name='test_cost')
	test_cost_summary = tf.summary.scalar('test cost', test_cost)

	evolution_summaries = tf.summary.merge([train_cost_summary,test_cost_summary])

	#####

	init = tf.global_variables_initializer()


	with tf.Session() as sess :
		writer = tf.summary.FileWriter(os.path.join('graphs'), sess.graph)
		sess.run(init)
		prev_train_cost = 0
		epoch_train_cost = 100
		epoch = 0
		#np.abs((prev_train_cost - epoch_train_cost))  > 0.000000001 and
		while epoch <max_epoch:
			epoch +=1
			prev_train_cost = epoch_train_cost
			sess.run(optimizer,feed_dict={X:X_train, Y:Y_train})

			epoch_train_cost = cost.eval({X: X_train, Y: Y_train})
			epoch_test_cost = cost.eval({X: X_test, Y: Y_test})
			summ = sess.run(evolution_summaries,feed_dict={train_cost: epoch_train_cost, test_cost: epoch_test_cost})

			writer.add_summary(summ, epoch)

			if epoch % 5 == 0: print("Cost before epoch {}: {}".format(epoch, epoch_train_cost))

		final_train_cost = cost.eval({X: X_train, Y: Y_train})
		final_test_cost = cost.eval({X: X_test, Y: Y_test})
		print('Final train cost : {} \nFinal test cost : {}'.format(final_train_cost,final_test_cost))

		predictions_test = Z3.eval({X: X_test, Y: Y_test})
		print('Extract of predictions test')
		[print('Prediction : {} Price : {}'.format(int(predictions_test[0,i]),Y_test[0,i])) for i in range(10)]

		Y_submit = Z3.eval({X:X_Submit})
		write_submission(Y_submit,X_Submit)


if __name__=='__main__' :
	X_train, X_test, y_train, y_test, X_submit = load_preprocess_datas()
	describe_datasets(X_train, X_test, y_train, y_test, X_submit)
	model(X_train,y_train,X_test,y_test,X_submit,max_epoch=100000)



