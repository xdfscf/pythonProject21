class VILTConfigure():
    model_name = "vilt"
    def __init__(self,
                 vit_hidden_size=768,
                 vit_intermediate_size=1024,
                 vit_attention_heads=4,
                 vit_hidden_dropout_prob=0.0,
                 vit_attention_dropout_prob=0.0,
                 bert_vocab_size = 30522,
                 bert_hidden_size= 768,
                 bert_attention_heads= 4,
                 bert_intermediate_size=1024,
                 bert_max_position_embeddings=50,
                 bert_type_vocab_size=2,
                 bert_hidden_dropout_prob=0.0,
                 bert_attention_probs_dropout_prob=0.0,
                 vilt_hidden_size=768,
                 vilt_intermediate_size=1024,
                 vilt_attention_heads=4,
                 vilt_hidden_dropout_prob=0.0,
                 vilt_attention_dropout_prob=0.0,
                 modality_type_vocab_size=2,
                 num_layers=6,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 image_size=224,
                 image_padding=False,
                 patch_size=16,
                 img_channels=3,
                 qkv_bias=True,
                 label_size=78,
                 **kwargs,
                 ):
        self.vit_hidden_size = vit_hidden_size
        self.vit_intermediate_size=vit_intermediate_size
        self.vit_attention_heads = vit_attention_heads
        self.vit_hidden_dropout_prob = vit_hidden_dropout_prob
        self.vit_attention_dropout_prob = vit_attention_dropout_prob
        self.bert_vocab_size = bert_vocab_size
        self.bert_hidden_size = bert_hidden_size
        self.bert_attention_heads = bert_attention_heads
        self.bert_intermediate_size = bert_intermediate_size
        self.bert_max_position_embeddings = bert_max_position_embeddings
        self.bert_hidden_dropout_prob = bert_hidden_dropout_prob
        self.bert_attention_probs_dropout_prob = bert_attention_probs_dropout_prob
        self.bert_type_vocab_size=bert_type_vocab_size
        self.vilt_hidden_size=vilt_hidden_size
        self.vilt_intermediate_size=vilt_intermediate_size
        self.vilt_attention_heads=vilt_attention_heads
        self.vilt_hidden_dropout_prob=vilt_hidden_dropout_prob
        self.vilt_attention_dropout_prob=vilt_attention_dropout_prob
        self.modality_type_vocab_size=modality_type_vocab_size
        self.num_layers = num_layers
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.image_padding = image_padding
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.qkv_bias = qkv_bias
        self.label_size = label_size
