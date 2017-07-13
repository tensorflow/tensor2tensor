
```python

def unmagic_encoder(encoder_input,
                    hparams,
                    name="encoder"):
  x = encoder_input
  
  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  
  with tf.variable_scope(name):
    pass
  return x
  
def magic_decoder(decoder_input,
                  encoder_output,
                  residual_fn,
                  encoder_self_attention_bias,
                  decoder_self_attention_bias,
                  encoder_decoder_attention_bias,
                  hparams,
                  name="decoder"):
  x = decoder_input
  y = encoder_output
  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="decoder_self_attention"))
        with tf.variable_scope("enc"):
          y = residual_fn(
              y,
              common_attention.multihead_attention(
                  y,
                  None,
                  encoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  summaries=summaries,
                  name="encoder_self_attention"))
          y = residual_fn(y, transformer.transformer_ffn_layer(y, hparams))
        
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                y,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="encdec_attention"))
        x = residual_fn(x, transformer.transformer_ffn_layer(x, hparams))
  return x
```
