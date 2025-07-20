import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class PSR(SequentialRecommender):
    r"""
    PSR is learning dual-level representations for sequential recommendation.

    """

    def __init__(self, config, dataset):
        super(PSR, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        
        #load patch parameters
        self.patch_length = config['patch_len']
        self.patch_stride = config['stride']
        self.padding_patch = config['padding_patch'] 
        self.patch_lambda = config['patch_lambda']
        self.patch_fusion = config['patch_fusion']
        self.contrast = config['contrast']
        
        # Contrastive Learning
        self.lmd = config['lmd'] # contrastive loss weight
        self.tau = config['tau'] # temperature parameter
        self.sim = config['sim'] # similarity function type
        self.dropseq = config['dropseq'] 
        self.batch_size = config['train_batch_size']
        
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        
        
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        if self.padding_patch == 'end':
            self.patch_num = int((self.max_seq_length - self.patch_length)//self.patch_stride) + 1
        elif self.padding_patch == 'repeat':
            self.patch_num = int((self.max_seq_length - self.patch_length)//self.patch_stride) + 2

        self.input_linear = nn.Linear(self.patch_length * self.hidden_size, self.hidden_size)
        
        if self.patch_fusion == 'linear':
            self.linear = nn.Linear(self.patch_num * self.hidden_size, self.hidden_size)
        elif self.patch_fusion == 'last':
            self.linear = nn.Linear(self.patch_num, self.max_seq_length)
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_attention_mask_patch(self, patches):
        """
        Generate attention mask for the patched input sequence.

        Args:
            patches (torch.Tensor): The patched input sequence, shape [batch_size, num_patches, patch_length].

        Returns:
            extended_attention_mask (torch.Tensor): The attention mask, shape [batch_size, 1, num_patches, num_patches].
        """
        attention_mask = (patches.sum(dim=-1) > 0).long()  
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_patches]
        subsequent_mask = torch.triu(torch.ones((1, attention_mask.size(-1), attention_mask.size(-1))), diagonal=1).to(patches.device)
        subsequent_mask = subsequent_mask.long()
        
        extended_attention_mask = extended_attention_mask * (1 - subsequent_mask)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # Large negative value for masking

        return extended_attention_mask

    def patching(self, input_seq):
        """
        This function divides a sequence into patches.
        It repeats the last patch and appends it.
        
        Args:
            input_seq: Input sequence with shape [batch_size, seq_len] or [batch_size, seq_len, hidden_size]
            
        Returns:
            patches: Sequence divided into patches with shape [batch_size, num_patches + 1, patch_length] or 
                    [batch_size, num_patches + 1, patch_length, hidden_size]
        """
        seq_len = input_seq.size(1)
        
        # 필요한 패딩 길이 계산
        num_patches = (seq_len - self.patch_length) // self.patch_stride + 1
        total_length = (num_patches - 1) * self.patch_stride + self.patch_length
        pad_len = total_length - seq_len
    
        if pad_len > 0:
            if input_seq.dim() == 2:  # [batch_size, seq_len]
                pad_tensor = torch.zeros(input_seq.size(0), pad_len, device=input_seq.device, dtype=input_seq.dtype)
            elif input_seq.dim() == 3:  # [batch_size, seq_len, hidden_size]
                pad_tensor = torch.zeros(input_seq.size(0), pad_len, input_seq.size(2), device=input_seq.device, dtype=input_seq.dtype)
            input_seq = torch.cat([input_seq, pad_tensor], dim=1)
            
        patches = input_seq.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        
        if self.padding_patch == 'end':
            pass
        elif self.padding_patch == 'repeat':
            last_patch = patches[:, -1:, ...] 
            patches = torch.cat([patches, last_patch], dim=1) 

        return patches

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        
        #item_emb
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb   
        
        # Get device for this layer's operations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.item_embedding.weight.device
        
        ########indivisual item########
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        #########patching item#########      
        patch_seq = self.patching(item_seq).to(device)

        patch_seq = patch_seq.to(torch.float32)  #
        input_patch_emb = self.patching(item_emb) # [batch_size, num_patches + 1, hidden_size, patch_length]
        bs, n_patch, _, _ = input_patch_emb.shape
        input_patch_emb = input_patch_emb.reshape(bs, n_patch, -1)
        input_patch_emb = self.input_linear(input_patch_emb)
        input_patch_emb = self.LayerNorm(input_patch_emb)
        input_patch_emb = self.dropout(input_patch_emb)
        
        extended_attention_mask = self.get_attention_mask_patch(patch_seq)
        
        trm_output = self.trm_encoder(
            input_patch_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        ### patch output ###
        output_patch = trm_output[-1]  # [batch_size, num_patches, hidden_size]

        if self.patch_fusion == 'mean':
            output_patch = output_patch.mean(dim=1)
        elif self.patch_fusion == 'sum':
            output_patch = output_patch.sum(dim=1)
        elif self.patch_fusion == 'last':
            output_patch = output_patch.permute(0, 2, 1)
            output_patch = self.linear(output_patch) # [batch_size, n_items, hidden_size]
            output_patch = self.dropout(output_patch) 
            output_patch = output_patch.permute(0, 2, 1)
            output_patch = self.gather_indexes(output_patch, item_seq_len - 1)
        else: # linear
            output_patch = output_patch.reshape(output_patch.size(0), -1)
            output_patch = self.linear(output_patch)
            output_patch = self.dropout(output_patch)

        return output, output_patch

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, patch_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type == 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            logits_patch = torch.matmul(patch_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items) + self.patch_lambda * self.loss_fct(logits_patch, pos_items)
            
        
        if self.contrast == 'us':            
            if self.dropseq:
                seq_output = self.dropout(seq_output)            
            
            nce_logits, nce_labels = self.info_nce(seq_output,
                                                    patch_output,
                                                    temp=self.tau,
                                                    batch_size=item_seq_len.shape[0],
                                                    sim=self.sim
                                                )
            nce_loss = self.nce_fct(nce_logits, nce_labels)
            
            return loss + self.lmd * nce_loss
        
        else:
            return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.cdist(z, z, p=2)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()
    
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
    
        return alignment, uniformity

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores