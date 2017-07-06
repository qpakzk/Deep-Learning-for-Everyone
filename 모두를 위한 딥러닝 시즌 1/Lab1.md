
# Lab 1

## Tensorflow Hello World!


```python
import tensorflow as tf
hello = tf.constant("Hello, TensorFlow")

sess = tf.Session()

print(sess.run(hello))
```

    b'Hello, TensorFlow'


## Computational Graph


```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
```


```python
print("node1:", node1, "node2:", node2)
print("node3:", node3)
```

    node1: Tensor("Const_1:0", shape=(), dtype=float32) node2: Tensor("Const_2:0", shape=(), dtype=float32)
    node3: Tensor("Add:0", shape=(), dtype=float32)



```python
sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))
```

    sess.run([node1, node2]): [3.0, 4.0]
    sess.run(node3): 7.0


## Placeholder


```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
```

    7.5
    [ 3.  7.]

