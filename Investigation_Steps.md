#AI 

- [ ] **AI Modeling** [[AI + Meca]]
	- [ ] Dog/Cat discrimination A-Z classical
		- [ ] 🐇🕳️ rabbit hole: https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-10-16-01-Image-Classification-with-Cat-and-Dog.ipynb
		- [ ] https://www.kaggle.com/code/orensa/cats-vs-dogs-image-classification-model
		- [ ] accuracy of 100% ?: https://www.kaggle.com/code/asif00/exploring-cnn-models-cats-and-dogs-classification
		- [ ] very good model: https://github.com/sancharika/Dog-Cat-Classification/blob/main/Cat%20dog.ipynb
		- [ ] preprocess: https://www.tensorflow.org/tutorials/load_data/images?hl=fr
		- [ ] Model evaluation Dashboard
			- accuracy
			- Loss
			- Time
			- Roc Curve
		- [ ] Model vizualisation
			- Model Architechture 
		- [ ] Functions to create:
			- [ ] show_false_prediction (afficher les résultat faux dans l'evaluation)
			- [ ] visualize_validation_results & visualize_train_results
			- [x] Function to keep all the features and the result in a `.csv`file
			- [ ] Create Co-Incidence [[Cognitive_AI_Librairie]]
		- [ ] Create function to create report automatically 
			- [ ] for vizualization: https://plotly.com/python/roc-and-pr-curves/
		- [ ] Create a generelized version of the code to easy do Grid search
			- [ ] Optimizer: https://keras.io/api/layers/
			- [ ] Layers: https://keras.io/api/layers/
		- [ ] Create a model with a simple NN architecture, Normal Architecture and Complex Architecture
		- [ ] Create a model to predict the accuracy where the parameters are input the features (X) and where the accuracy is the predicted features (Y)
			- [ ] Plot it, And what is the value of $R^2$ ?
		- [ ] ❓ create the complete code in layer even the data preprocessing 
			- [ ] https://keras.io/api/layers/preprocessing_layers/
			- [ ] https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/preprocessing_layers.ipynb
	- [ ] How is the size of the image is affecting the accuracy 
		- [ ] How having more or less images is affecting the model?
		- [ ] With a very good model, a normal model and a dumb model
	- [ ] How the number of images of `cat` or `dog` is affecting the accuracy
	- [ ] How the splitting rate is affecting the accuracy
	- [ ] How Black & White image is affecting accuracy
		- [ ] Add a red triangle to each cat picture and a green square to each dog picture. train the model then use the model on a normal Test set.
		- [ ] Is training a model with blur images give a better accuracy on Validation images?
		- [ ] McCollough effect ? https://fr.wikipedia.org/wiki/Effet_McCollough
		- [ ] Mc Gurk effect ? https://fr.wikipedia.org/wiki/Effet_McGurk
	- [ ] Data augmentation
		- [ ] How is data augmentation is affecting the accuracy 
	- [ ] Dog/Cat discrimination + NEAT algorithm (Deep Neuro Evolution Approach)
		- [ ] https://arxiv.org/pdf/2112.07057v1.pdf
		- [ ] https://arxiv.org/pdf/2211.16978.pdf
		- [ ] https://arxiv.org/pdf/2309.12148.pdf
		- [ ] https://www.youtube.com/watch?v=h9JZ0YHtKWQ
		- [ ] https://github.com/harvitronix/neural-network-genetic-algorithm
		- [ ] https://github.com/jliphard/DeepEvolve
		- [ ] with tensorflow: https://github.com/PaulPauls/Tensorflow-Neuroevolution
		- [ ] Uber research: https://github.com/uber-research/PyTorch-NEAT
		- [ ] Map Elite: https://github.com/uber-research/Map-Elites-Evolutionary
	- [ ] Dog/Cat discrimination Highest score possible
	- [ ] Unsupervised clustering on Dog/cat Dataset
		- [ ] itérer afin jusqu'a ce que l'accuracy soit bonn (j'imagine que l'accuracy ne peut pas etre bone si il n'y a pas le bon nombre de cluster/label)
		- [ ] https://colab.research.google.com/drive/1FmZ_fBmvhnm1FXjwi5BwLvzkh9XtzpqT?authuser=1#scrollTo=pzLYfqmcbtCP
		- [ ] Etre capable d'afficher les contenu des cluster
		- [ ] créer automatiquement les fichiers des labels en fonction des cluster 
	- [ ] Try Dog/Cat discrimination with Graph Neural Network (GNN)
	- [ ] Dog/Cat discrimination on sounds of the animal + images
		- [ ] https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs
		- [ ] https://www.tensorflow.org/tutorials/audio/simple_audio?hl=fr
		- [ ] https://www.tensorflow.org/tutorials/audio/transfer_learning_audio?hl=fr
	- [ ] Train an Auto encoder from the Dog/Cat dataset and do discrimination from the output of the Autoencoder
		- [ ] https://keras.io/examples/generative/vae/
		- [ ] https://www.kaggle.com/code/rvislaywade/visualizing-mnist-using-a-variational-autoencoder/notebook
		- [ ] https://www.siarez.com/projects/variational-autoencoder
		- [ ] https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
		- [ ] https://blog.keras.io/building-autoencoders-in-keras.html
	- [ ] Train a Generator and a Discriminator (GAN)
		- [ ] https://www.geeksforgeeks.org/generative-adversarial-network-gan/
		- [ ] https://keras.io/examples/generative/dcgan_overriding_train_step/
	- [ ] Classification with Energy based Modeling
	- [ ] Petit robot (rasp, camera, motor): quand il reconnaît un chat il dit chat, quand il reconnaît un chien il dit chien 
	- [ ] Applly a human bias on:
	- For exemple put a set of images of human faces, and train the AI to discriminate if the person is a good or a bad person (with different sensory on: image, text, voice, high etc)
	- [ ] Adversarial Attack on dogs and cats Model
		- [ ] https://github.com/etotheipi/toptal_tensorflow_blog_post/blob/dev/adversarial_example/adversarial_cats_dogs.ipynb
	- [ ] ❓ Reinforcement Learning? 
		- [ ] https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic?hl=fr
		- [ ] https://www.tensorflow.org/agents?hl=fr
	- [ ] ❓ Q-Learning ?
		- [ ] https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=fr
	- [ ] ❓ Active Learning?
	- [ ] ❓Agent Based model
		- [ ] https://en.wikipedia.org/wiki/Agent-based_model
		- [ ]  https://www.tensorflow.org/agents?hl=fr

❓Deploy the model on IOs app
❓Create an api with keras: https://blog.keras.io/

- [ ] **loool Quantum AI:**
	- [ ] https://quantumai.google/cirq/experiments/qaoa/optimization_analysis
	- [ ] https://blog.tensorflow.org/2021/06/training-with-multiple-workers-using-tensorflow-quantum.html?hl=fr
	- [ ] https://www.youtube.com/watch?v=-o9AhIz1uvo


**❓Questions**
>- Check overfitting 
>- How much changing the color modifing the accuracy ?
>- How each features is affecting the $Per$ and tha $Cog$?
>- What are the most important features ?
>-  Supermodularity verification
>- Can we calcul wich transformation should we do on images to optimize the accuracy

### To implement
**Plot Neural Architecture**  
- Plot Neural Net Architecture: [http://alexlenail.me/NN-SVG/AlexNet.html](https://colab.research.google.com/corgiredirector?site=http%3A%2F%2Falexlenail.me%2FNN-SVG%2FAlexNet.html)  
- computational graph models: [https://www.graphcore.ai/posts/what-does-machine-learning-look-like](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.graphcore.ai%2Fposts%2Fwhat-does-machine-learning-look-like)  
- Plot Learning: [https://github.com/graphcore](https://github.com/graphcore)  

**Data selection**
- Find the most important Features: [https://github.com/AxeldeRomblay/MLBox/blob/master/examples/classification/example.ipynb](https://github.com/AxeldeRomblay/MLBox/blob/master/examples/classification/example.ipynb)  

**Advenced AI**
cours de LeCun: https://atcold.github.io/NYU-DLSP20/fr/

- Energy based modeling: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
- Graph Neural Network: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
- Transformers and attention: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- Meta learning: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html
- Self supervised learning: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html



---

**Backup references**
- *TP_N2:* https://colab.research.google.com/drive/19Vnf-26-EtKpg8wa819fvH6afCaM9stS#scrollTo=yiyFTTndsHWo
- *TP BigData:* https://colab.research.google.com/drive/1jCKbRWko0q41Yp2S_wNvykueIb2vQnuC
- *TP Clustering:* https://colab.research.google.com/drive/1hOnhS0Rsjn04ixt8jtlzlF1BO1hKD2jD
- *TP Deep Learning :* https://colab.research.google.com/drive/1zfyIbAdZsbb-EdkE3LFCwXVjYbxWYfqN#scrollTo=A-TivddwoQrQ
- *DeepNeuro Evolution du futur:* https://colab.research.google.com/drive/1Hy_RGlJ_qYK9UbzDyZPj3A6fw10puqwF?authuser=1
- *NLP:* https://colab.research.google.com/drive/1blO8yaGV1oESiWmNbZC31S7z7oArkxVB?authuser=1#scrollTo=CqHz8wDLmz8A
- *Decision Tree & Random Forest:* https://colab.research.google.com/drive/1M5JkZZ4Y7JdBdtpaEgM_X53rfMtrVF8O?authuser=1
- *K Means:* https://colab.research.google.com/drive/1z3cETArrDKzjyWvZ4A-qsQO5hQkOqGPW?authuser=1
- *Neural Networks Fundamentals:* https://colab.research.google.com/drive/1aR5EuYbiBzuyWwP3jUyE7_CEEeBokpu_?authuser=1
- *Deep Neural Network:* https://colab.research.google.com/drive/1OrsSP4lHbMlPnthI80mOg_aXwfZQAb0i?authuser=1
- *CNN:* https://colab.research.google.com/drive/167wuvYDKT9IWY25WDED-yFB9zrtjcKU1?authuser=1
- *RNN:* https://colab.research.google.com/drive/1VLWUxa_ZmUg-sj7xCZeAGYsT_6W_johY?authuser=1#scrollTo=Rrjv5hRs2ypz
- *RNN-LSTM:* https://colab.research.google.com/drive/1vnhQ7763VLGAtS9hjjqBiuUaYCW1F80I?authuser=1
- *VAE:* https://colab.research.google.com/drive/1FmZ_fBmvhnm1FXjwi5BwLvzkh9XtzpqT?authuser=1
- *Reinforcement Learning:* https://colab.research.google.com/drive/1CPjZb8B54PoYRHaeEhsUhCmNaDsCJvR2?authuser=1
- Excellent ressources en IA: https://github.com/bnsreenu/python_for_microscopists
- Graph Viz: https://colab.research.google.com/drive/13hYkprRaXBqCUwzigSwlnfns0xmh2A1h
- Segmentation Dog,Cat: https://colab.research.google.com/drive/1OKdmlh-czoIk-v2FLPUUH1TYOdHM4AxZ?authuser=1#scrollTo=Q0rIzvN1cy8H
- Grid Search exemple: https://colab.research.google.com/drive/13Cvg7toc0vFSQjMN0lT9SBp2yzR-vMUP?authuser=1

---
## Main parameters

```python
all_possible_genes = {
#-------------------------
#     DATA PREPROCESSING 
#-------------------------
'nb_images_dog_reduction' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
'nb_images_cat_reduction' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
'Black&White_Train':[0,1]
'Black&White_Validation':[0,1]
'Blur_Train':[Yes, No]
'Blur_Validation':[Yes, No]
'image_height' :  #variation of the image croping
'image_width' :
'learning_support': [Yes, No] #petite pastille pour apprendre (pour entrainement chien chat, validation etc)

#-------------------------
#     DATA AUGMENTATION
#-------------------------
#precision of the types of data augmentation
'Data_Augmentation': [Yes, No]
#-------------------------
#     SPLIT TRAINING SET AND VALIDATION SET
#-------------------------
'Splitting_Rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#---------------------------------------
#      MODEL ARCHITECTURE DESIGN
#---------------------------------------
'nb_neurons': [16, 32, 64, 128],
'nb_layers': [1, 2, 3, 4, 5],
'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],

#---------------------------------------
#       MODEL COMPILING
#---------------------------------------
'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
'learning_rate':[]

#---------------------------------------
#       MODEL TRAINING
#---------------------------------------
'EPOCHS': [5, 10, 15, 20, 30, 35]
'BATCH_SIZE' : [15, 30]
}
```
