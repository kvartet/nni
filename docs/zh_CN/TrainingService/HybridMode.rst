**以混合模式进行实验**
===========================================

在混合模式下运行 NNI 意味着 NNI 将在多种培训平台上运行试验工作。 例如，NNI 可以同时将试用作业提交到远程计算机和 AML。

设置环境
-----------------

对于混合模式，NNI 目前支持的平台有 `本地平台 <LocalMode.rst>`__\ ，`远程平台 <RemoteMachineMode.rst>`__\ ， `PAI <PaiMode.rst>`__ 和 `AML <./AMLMode.rst>`__\ 。 使用这些模式开始 Experiment 之前，用户应为平台设置相应的环境。 有关环境设置的详细信息，请参见相应的文档。

运行实验
-----------------

以 ``examples/trials/mnist-tfv1`` 为例。 NNI 的 YAML 配置文件如下：

.. code-block:: yaml

    experimentName: MNIST
    searchSpaceFile: search_space.json
    trialCommand: python3 mnist.py
    trialCodeDirectory: .
    trialConcurrency: 2
    trialGpuNumber: 0
    maxExperimentDuration: 24h
    maxTrialNumber: 100
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
    trainingService:
      - platform: remote
        machineList:
          - host: 127.0.0.1
            user: bob
            password: bob
      - platform: local

To use hybrid training services, users should set training service configurations as a list in `trainingService` field.  
Currently, hybrid support setting `local`, `remote`, `pai` and `aml` training services.
