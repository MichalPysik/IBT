from architectures import createNetwork
from keras.utils import plot_model

#Tabular_MLP = createNetwork('Tabular', 0, input_shape=(50,), num_classes=2)
#Tabular_CNN = createNetwork('Tabular', 1, input_shape=(50,), num_classes=2)
#Tabular_RNN = createNetwork('Tabular', 2, input_shape=(50,), num_classes=2)
#Tabular_MLPx = createNetwork('Tabular', 3, input_shape=(50,), num_classes=2)
#
#Image_MLP = createNetwork('Image', 0, input_shape=(28, 28), num_classes=10)
#Image_CNN = createNetwork('Image', 1, input_shape=(28, 28), num_classes=10)
#Image_RNN = createNetwork('Image', 2, input_shape=(28, 28), num_classes=10)
#Image_CNNx = createNetwork('Image', 3, input_shape=(28, 28), num_classes=10)
#
#Sequence_MLP = createNetwork('Sequence', 0, input_shape=(10000,), num_classes=2)
#Sequence_CNN = createNetwork('Sequence', 1, input_shape=(500,), num_classes=2)
#Sequence_RNN = createNetwork('Sequence', 2, input_shape=(500,), num_classes=2)
Sequence_RNNx = createNetwork('Sequence', 3, input_shape=(500,), num_classes=2)
#
#plot_model(Tabular_MLP, to_file='model_plots/Tabular_MLP.png', show_shapes=True, show_layer_activations=True)
#plot_model(Tabular_CNN, to_file='model_plots/Tabular_CNN.png', show_shapes=True, show_layer_activations=True)
#plot_model(Tabular_RNN, to_file='model_plots/Tabular_RNN.png', show_shapes=True, show_layer_activations=True)
#plot_model(Tabular_MLPx, to_file='model_plots/Tabular_MLPx.png', show_shapes=True, show_layer_activations=True)
#
#plot_model(Image_MLP, to_file='model_plots/Image_MLP.png', show_shapes=True, show_layer_activations=True)
#plot_model(Image_CNN, to_file='model_plots/Image_CNN.png', show_shapes=True, show_layer_activations=True)
#plot_model(Image_RNN, to_file='model_plots/Image_RNN.png', show_shapes=True, show_layer_activations=True)
#plot_model(Image_CNNx, to_file='model_plots/Image_CNNx.png', show_shapes=True, show_layer_activations=True)
#
#plot_model(Sequence_MLP, to_file='model_plots/Sequential_MLP.png', show_shapes=True, show_layer_activations=True)
#plot_model(Sequence_CNN, to_file='model_plots/Sequential_CNN.png', show_shapes=True, show_layer_activations=True)
#plot_model(Sequence_RNN, to_file='model_plots/Sequential_RNN.png', show_shapes=True, show_layer_activations=True)
plot_model(Sequence_RNNx, to_file='model_plots/Sequential_RNNx.png', show_shapes=True, show_layer_activations=True)