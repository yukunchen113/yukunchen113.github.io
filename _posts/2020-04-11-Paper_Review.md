---
layout: default
categories: [article]
title: "MONet"
---
Identify Article
- motivation
	- object decomposition helps with downstream tasks
		- graph-structured networks
			- reinforcement learning
			- physical modeling
			- multiagent control
- purpose
	- object decomposition
		- unsupervised identification of objects
	- same representation space for all objects
	- infer objects with occlusion
	- varying objects in scene
	- generalizable to:
		- number of objects
		- novel combintions of factors of variation (FOV)
		- objects that apear frequently together

- contributions
	- disentanglement of multiple objects
	- unsuperivised generative model

- advantages
	- current object segementation techniques don't learn object representations
	- GQNs can render 3D scenes, but they require multiple viewpoints
	- disentanglement only works with one object

- disadvantages
- method
	- attention network generates a mask, which the component vae makes predictions based on.
	- latents are created based on the mask
	- reconstruction loss of inputs is unconstrained in masked regions
	- each mask will represent a different component
	- VAE required to model the masks
		- probability of specific component given set of masks
		- specific components are modeled given the latents
	- use rnn to track parts that have not been attributed to a component yet
	- model is trained using normal Beta-VAE loss, with an extra loss term to distinguish between masking components.