# coding: utf-8

# # Lab 1

# ## Tensorflow Hello World!

# In[1]:


import tensorflow as tf
hello = tf.constant("Hello, TensorFlow")

sess = tf.Session()

print(sess.run(hello))


# ## Computational Graph

# In[2]:


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


# In[3]:


print("node1:", node1, "node2:", node2)
print("node3:", node3)


# In[4]:


sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))


# ## Placeholder

# In[5]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

