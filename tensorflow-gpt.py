import tensorflow as tf
import numpy as np
import scipy.io

# Load the pre-trained model
model_path = 'imagenet-vgg-verydeep-19.mat'
model = scipy.io.loadmat(model_path)

# Define the content and style images
content_image_path = 'MIKU.jpg'
style_image_path = 'p0.jpg'

# Preprocess the images to be compatible with the model
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    img_shape = image.size
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image, img_shape

content_image, content_shape = preprocess_image(content_image_path)
style_image, style_shape = preprocess_image(style_image_path)

# Define the layers for content and style representation
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Build the model for feature extraction and loss calculation
def get_model(layer_names):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model

def gram_matrix(input_tensor):
    flat_tensor = tf.reshape(input_tensor, (-1, input_tensor.shape[-1]))
    n = tf.shape(flat_tensor)[0]
    gram = tf.matmul(flat_tensor, flat_tensor, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = get_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        return {'content': content_dict, 'style': style_outputs}

extractor = StyleContentModel(style_layers, content_layers)

# Define the loss functions
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_loss = tf.add_n([tf.reduce_mean((output - target) ** 2)
                           for output, target in zip(outputs['style'], style_targets)])
    style_loss *= style_weight / extractor.num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((output - target) ** 2)
                           for output, target in zip(outputs['content'].values(), content_targets)])
    content_loss *= content_weight / len(content_layers)

    loss = style_loss + content_loss
    return loss

# Define the optimizer and metrics
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight=1e-2
content_weight=1e4
total_variation_weight=30

# Define the iteration loop to generate the new image
def style_content_optimization(content_image, style_image, num_iterations=1000):
    input_image = tf.Variable(content_image)
    inputs = tf.keras.applications.vgg19.preprocess_input(input_image)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            outputs = extractor(inputs)
            loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
            loss += total_variation_weight * tf.image.total_variation(input_image)
        grad = tape.gradient(loss, input_image)
        opt.apply_gradients([(grad, input_image)])
        inputs.assign(tf.keras.applications.vgg19.preprocess_input(input_image))
        if i % 100 == 0:
            print('Iteration %d/%d - loss: %f' % (i+1, num_iterations, loss))
    final_image = input_image.numpy()
    final_image = tf.keras.preprocessing.image.array_to_img(final_image[0])
    return final_image

# Generate the new image
result_image = style_content_optimization(content_image, style_image, num_iterations=500)

# Save the generated image
result_image.save('result_image.jpg')
