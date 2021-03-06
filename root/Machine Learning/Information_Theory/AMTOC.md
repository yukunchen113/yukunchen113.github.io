---
layout: default
title: A Mathematical Theory of Communcation
permalink: /ml/information_theory/AMTOC/
---
# A Mathematical Theory of Communcation
By Claude Shannon. In these notes, we will go over this paper. I hope to provide summaries to new points presented, as well as provide some of my own thoughts as they happen. My own thoughts could be wrong, as I am just brain storming so please let me know if you disagree with one of my points. Please keep in mind that this is an old paper, and some of the examples that it presents might be a little outdated (the theories should still stand.)

## Introduction
- What is modulation, PCM and PPM? It seems that these methods allow for a tradeoff between noise in the signal and bandwith.
- paper presents the effect of noise on the transmission, and how to leverage statistical properties in your system. 
	- So here is the main idea in the paper and everything should connect back to and relate to these ideas.


- _Fundamental problem of communication_: to reproduce a message from one point to another point. The points can be different points in space or time.
	- messages and be recreated exacty or approximately. 
		- To reproduce a message exactly might be difficult or impossible sometimes, due to a lack of memory or medium to capture or store the information. However, we might not need to get it exactly. If it sufficiently reproduced in an efficent manner, we might want to trade off some accuracy for this efficient reproduction of the message.
	- what defines a _message_?
		- a message is a representation of information.
- Messages usually have meaning/purpose behind them. This meaning is not important as it will not affect the transmission of the signal. This means that once we have a sufficient understanding of the behavior of the system (statistically) and also a proper embedding for the items to transmit, then the meaning no longer remains important. Just the message and the possible set of messages which it came from, is important.
- Also we must optimize across the set of all possible messages, rather than just one message. This will allow for gains in the long run, as the messages are shown. We also don't know which specific message will be sent.


- when will the number of messages in the set be infinite?
	- number of messages could be infinite when we have an infinite number of choices to make. The only case where it is infinite is where we would have a continuous number of choices. For example if you wanted to decide whether to pick up a blue pen or a red pen, you also have the choice to move your hand to pick up something else, or you can also have the choice of moving your had 3 cm to the right (with out picking up anything) or 3.1 cm to the right, etc. This action might not make sense to make, but it is still a choice that you can take. We can go choose an infinite amount of distances of how much you can choose to just move your had by. But, why do we not take these into consideration? Because they have a very low chance of happening since right now you are looking at which pen you would choose. To decrease the amount of messages that we would have to represent, as well as, eliminate useless information, we could split the messages to be: blue pen, red pen, and neither. We don't need the exact details of the neither, if we just care about the pens.
- why does a monotonic function of the number of messages in the set, produce the same result as that number?
	- I am assuming that the actual value of the number is arbitrary, as long as we have a standard process of defining that number. Being a monotonic function, it will allow for a uniqueness between each actual value.
- what is information? How does the number of messages = information when one messge is selected? (when all messages are equally likely)
	- information is something that will decrease the uncertainty pertaining to the messages in your set.
	- What about information that could increase/doesn't change uncertainty? For example, if you had a problem where you needed to find what X is. X is equally likely in being A,B or C. Then, I tell you information that X can also be D. So now, you know that X can be A,B,C or D. This information brought in D, but it didn't help towards figuring out what X is. Remember our definition above. That we have constructed a system/set of messages already. This information that tells us about D is trying to expand our set of messages. This is not what we are looking for. This just means that you have defined the requirements for your system incorrectly. That is why information in our case, is to decrease the uncertainty in the current set of messages.


- why is logorithmic the most natural choice? For the monotonic function.
- usually in engineering, we see that adding a new component will increase log of information that we could represent linearly.
	- eg. if we have a 7-bit register vs an 8 bit register, we would have twice as many possible states in the 8 bit case. An $N$ amount of increase in bits will increse the number of representations by $2^N$, or equivalently if we used the log(number of representations), we would have $Nlog_22=N$ representations. Here, the number of log-representations is growing linearly.
- using log os also better when working with the math. Using base $e$ will be clean to solve.
- If we use log base 2, then we would be using _bits_ as a unit of measurement for information.

## Discrete Noiseless Systems
### The Discrete Noiseless Channel
we will be looking at how to measure the capacity of a discrete system to be able to transmit information.

What defines a discrete system? 
	- a message is a sequence that comes from a finite set of symbols.
		- eg. a bit, or dot/dash in telegraphy
	- sequence across time.
		- each element in the sequence may take varying amounts of time.
		- eg. dot takes 1 timestep, dash takes 3.
	- not all combinations of sequences need to be accounted for,
		- eg. if we want representations for 3 things, we won't need the complete set of possible sequences.

How do we define the capacity of a discrete channel? We need a general way to quantify sequences of different length and time. We can measure the capacity as the amount of information sent per unit of time. 

Remember that the amount of information is quantified by the amount of uncertainty solved. The total number of signals/sequences that we could be sending in a given amount of time is what we are uncertain about. Normalizing this with time would give us a general measure of the information in the given time.

- it seems that N(t) is equal to the sum of the $N(t-t_i)$, i = 1...n. We can think of each of these Ns recursively, as each of the $N(t-t_i)$ represents a sequence where the ending is the symbol $S_i$.
- what is the "well known result in finite differences"? 

TBD