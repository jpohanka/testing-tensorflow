# Java-hello-world

This is a simple hello-world-style program which is aimed to verify the proper functioning of the installed TensorFlow on your computer. 

The source code is taken from https://www.tensorflow.org/install/install_java

My setup for example:

* Linux distribution : Ubuntu 14.04
* TensorFlow 1.5


To run the maven build, type into the console the following line:
```
mvn -q compile exec:java
```

In some cases, the build may fail due to the following error:
```
An exception occured while executing the Java class. /tmp/tensorflow_native_libraries-1517435407546-0/libtensorflow_jni.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /tmp/tensorflow_native_libra
ries-1517435407546-0/libtensorflow_jni.so) -> [Help 1]
```

This is due to an out-dated version of `libstdc++`. The following command lists all available versions of `GLIBCXX` on your computer:
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

In order to install the latest version of `libstdc++` on Ubuntu, run the following commands:
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
```

Futhermore, in some cases the build may fail due to another error:
```
An exception occured while executing the Java class. /tmp/tensorflow_native_libraries-1517436589362-0/libtensorflow_jni.so:
/lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.23' not found (required by /tmp/tensorflow_native_libraries-15174365
89362-0/libtensorflow_jni.so)
```

This is due to an out-dated version of `libm.so`. Currently I have used a quickfix by using an older version TensorFlow 1.4 for the Java compilation.