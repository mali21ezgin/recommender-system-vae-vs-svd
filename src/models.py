import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from keras import ops
from keras import random
from surprise import SVD
from src.metrics import masked_mse

# Yeniden parametrelendirme ve KL kaybı için SamplingAndKL katmanı
class SamplingAndKL(Layer):
    
    def __init__(self, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        
    def call(self, inputs):
        z_mean, z_log_var = inputs  # Gizli dağılım ortalaması mu, log(varyansın karesi)
        
        batch = ops.shape(z_mean)[0]  # Gizli katmandaki ortalama değerdeki tensörün ilk indisi
        dim = ops.shape(z_mean)[1]
        
        epsilon = random.normal(shape=(batch, dim))
        
        kl_loss = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
        
        self.add_loss(self.beta * ops.mean(kl_loss))  # Toplam kayıp = Yeniden Yapılandırma Kaybı + (Beta * KL Kaybı)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon  # z = mu + std_dev * epsilon


def build_svd_model(n_factors=300, reg_all=0.05, random_state=42):
    """
    Surprise SVD (Matris Faktörizasyonu) Modeli Oluşturur
    """
    return SVD(n_factors=n_factors, reg_all=reg_all, random_state=random_state, verbose=False)


def build_vae_model(original_dim, latent_dim=50, intermediate_dim=200, dropout_rate=0.4, beta=0.01, learning_rate=0.0005):
    """
    Varyasyonel Otokodlayıcı (VAE) Modeli Oluşturur ve Derler
    """
    input_layer = Input(shape=(original_dim,), name='encoder_input')
    h = Dropout(dropout_rate)(input_layer)
    h = Dense(intermediate_dim, activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    z = SamplingAndKL(beta=beta, name='sampling_kl')([z_mean, z_log_var])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_output = Dense(original_dim, activation=None)

    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_output(h_decoded)

    vae = Model(input_layer, x_decoded_mean)
    optimizer = Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer, loss=masked_mse)
    
    return vae
