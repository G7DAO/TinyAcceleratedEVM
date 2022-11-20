# TinyAcceleratedEVM - A tiny EVM simulator designed for GPU/TPU parallelization

## Summary

A [gymnax](https://github.com/RobertTLange/gymnax) based EVM simulator written in the JAX machine learning framework. The idea is to write an EVM simulator that can be parallelized and gain significant performance boosts on the accelerated hardware such GPU/TPU. 

## Problem

There is no implementation of the EVM that can be parallelized across a GPU/TPU, and all implementations are based on CPU-based VMs. This means that there is a fundamental bottleneck to scale at which the EVM can be run on for simulation purposes, from mass-backtesting, simulation, monitoring, machine learning applications, etc in the future.

## Solution

[gymnax](https://github.com/RobertTLange/gymnax) is a framework under JAX, to write custom game environment that can be parallelized across accelerated hardware. 

The solution will consist of a very stripped down version of the EVM that can be parallelized across the GPU, with a small use-case. The idea isn't to implement a full, complete EVM for running as an actual node, but rather to build a testing environment/framework that can be used by developers.

The prototype will consist of the following functionality:
* A gymnax simulator that models the EVM with the following functionality: 
  * Input an example EVM bytecode and execute it
  * Be able to input a custom message call to the simulator and get a response


## Getting Started


* Use cases:
  * Unit/Backtesting at scale for certain limited types of test, with no/low storage requirements
  * Training ML models on the GPU/TPU that can use this simulator

## FAQ
**Who will use this?**
* ML Engineers who want to leverage EVM simulation on the GPU for training ML models to interact/verify their trained models
* Security engineers who want to test EVM-based exploits at scale, or find new exploits through a search method

**What problem does this really solve?**
* Lack of EVM implementations that can run on accelerated hardware - bottleneck to running computation at scale to test EVM behaviour

**Is this just another walled garden designed to monetize something that's freely available?**
* **No.** The entire gymnax is open sourced, and also there is nothing like it available.

**Ok.. then how do you make money?**
* There will be a hyperoptimized implementation of this GPU/TPU based EVM simulator with more features, and other enterprise features. (e.g. security script testing at scale, based on scripts that developers can deploy through our API, so they don't need their own GPU setup)

**How do I know the code works as expected?**
* tinyAcceleratedEVM code is open-source, and can be inspected by anyone.
* Unit tests can be written in order to verify the same behaviour as other implementations of the EVM

### tinyAcceleratedEVM Roadmap

- [x] tinyAcceleratedEVM PRFAQ Live
- [ ] tinyAcceleratedEVM MVP released at end of Sozu Haus hacker house
- [ ] tinyAcceleratedEVM presented at ETHVietnam
- [ ] tinyAcceleratedEVM mass-scaling tool built
- [ ] tinyAcceleratedEVM commercialized to help businesses
