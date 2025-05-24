Home
=====

.. image:: media/amago_logo_3.png
   :alt: amago logo
   :width: 310
   :align: center

.. centered:: **Adaptive RL with Long-Term Memory**

.. raw:: html

   <div align="center">
       <a href="https://arxiv.org/abs/2310.09971">
           <img src="https://img.shields.io/badge/Paper-AMAGO%20ICLR%202024-blue" alt="Paper #1">
       </a>
       <a href="https://arxiv.org/abs/2411.11188">
           <img src="https://img.shields.io/badge/Paper-AMAGO--2%20NeurIPS%202024-purple" alt="Paper #2">
       </a>
       <a href="https://ut-austin-rpl.github.io/amago">
           <img src="https://img.shields.io/badge/Docs-ut--austin--rpl.github.io%2Famago-4caf50" alt="Docs">
       </a>
   </div>

|
|

AMAGO follows a simple and scalable recipe for building RL agents that can generalize:

1. Turn meta-learning into a *memory* problem ("black-box meta-RL").
2. Put all of our effort into learning effective memory with end-to-end RL.
3. Treat other RL problems as special cases of meta-learning.
4. Use one method to solve a wide range of problems!

|

AMAGO is a high-powered off-policy version of `RL^2 <https://arxiv.org/abs/1611.02779>`_ for training large policies on long sequences. Its goal is to do this one thing very well, and make it applicable to just about any RL problem, while remaining customizable for research.

Some highlights:

- **Broadly Applicable**. Long-term memory, meta-learning, multi-task RL, and zero-shot generalization are special cases of its POMDP format. Supports discrete and continuous actions. Online and offline RL. See examples below!
- **Scalable**. Train large policies on long context sequences across multiple GPUs with parallel actors, asynchronous learning/rollouts, and large replay buffers stored on disk.
- **Easy to Modify**. Modular and configurable. Swap in your own model architectures, RL objectives, and datasets.

|
|


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   tutorial/index
   examples/index
   API Reference <api/amago>
   citation