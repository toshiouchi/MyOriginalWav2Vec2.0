'''
A feature of this class is to be able to calculate Lm without K.
Lm can be calculated without sampling about negative sample.
Lm is calculated for all cases about negative sample including positive sample.
This class is special, yet do not use in ordinary calculation.
In onrdinary caluculation K = min( num_targets_per_batch ) module was used.
-------------.
call usage
------------
wav2vec2loss = Wav2vec2Loss(
    contrastive_loss_temperature = 0.1,
    num_code_vector_groups = 2,
    num_code_vectors_per_group = 320,
    loss_alpha = 100.0,
)

batch_size = 8
time_sequence = 300
hidden_dim = 512

ouputs = torch.randn( 8, 300, 512 )
quantized_vector = torch.randn( 8, 300, 512 )
pgv_bar = torch.randn( 2, 320 )
mask = torch.randint(low=0, high=2, size=(8,300)).to( torch.bool )

# loss の計算など
loss, lm, ld, pos_sim, neg_sim = wav2vec2loss( outputs, quantized_vector, pgv_bar, mask )

'''

class Wav2vec2Loss(nn.Module):
    def __init__(self, 
        contrastive_loss_temperature = 0.1,
        num_code_vector_groups = 2,
        num_code_vectors_per_group = 320,
        loss_alpha = 0.1,
        ):
        super().__init__()
        self.k = contrastive_loss_temperature
        self.K = None
        self.Kmax = None
        self.cos = nn.CosineSimilarity(dim=-1)
        self.G = num_code_vector_groups
        self.V = num_code_vectors_per_group
        self.a = loss_alpha

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples, num_targets_mask, num_neg_sim_sum = self.negative_sampler(labels, num_targets_per_batch)

        contrastive_loss, pos_sim, neg_sim = self.contrastive_loss(target_encoder_out, labels, negative_samples, num_targets_mask, num_neg_sim_sum )
        diversity_loss = self.diversity_loss(perplexity)

        loss = contrastive_loss + self.a * diversity_loss

        return loss, contrastive_loss.item(), diversity_loss.item(), pos_sim, neg_sim

    def contrastive_loss(
            self,
            targets: torch.Tensor,
            labels: torch.Tensor,
            negative_samples: torch.Tensor,
            num_targets_mask,
            num_neg_sim_sum,
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            contrastive_loss torch.Tensor with shape `(1)`
            sim_mean schaler
            neg_sim_2mean_1mean schaler
        """

        sim = self.cos( targets, labels )
        similarity = torch.exp( sim / self.k)
        sim_mean = torch.mean( sim )

        neg_sim = self.cos( targets.unsqueeze(1), negative_samples )

        negative_similarity = torch.sum(torch.exp(( neg_sim / self.k)) * num_targets_mask, dim=1)
        neg_sim_sum = torch.sum( neg_sim * num_targets_mask, dim = 1 )

        neg_sim_1mean = ( neg_sim_sum - sim ) / ( num_neg_sim_sum - 1 )
        neg_sim_1mean_0mean = torch.mean( neg_sim_1mean, dim = 0 )
        contrastive_loss = -torch.log(similarity / negative_similarity).mean()

        return contrastive_loss, sim_mean.item(), neg_sim_1mean_0mean.item()

    def diversity_loss(self, perplexity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        log_perplexity = torch.log(perplexity)
        entropy = torch.sum(perplexity*log_perplexity, dim=-1)
        diversity_loss = torch.sum(entropy) / (self.G * self.V)

        return diversity_loss

    def negative_sampler(self, label: torch.Tensor, num_targets_per_batch: list[int]):
        """
        Args:
            label (torch.Tensor): with shape `(N, D)`
            num_targets_per_batch (list[int]): Number of targets per batch.

        Returns:
            negative_samples:torch.Tensor with shape `(N, Kmax, D)' inluding positive elements
            mask of negative_samples:torch.Tensor with shape `(N, Kmax, D )`
            number of negative_samples for each N with shape `(N)`

        """

        # Change from Original.  K != 100.  Kmax is max of num_targets_per_batch
        self.Kmax = max( num_targets_per_batch )

        negative_samples = torch.tensor( [], device=torch.device(device)  )
        num_targets_mask = torch.tensor( [], device=torch.device(device)  )
        num_neg_sim_sum = torch.tensor( [], device=torch.device(device)  )
        start_idx = 0

        for num_targets in num_targets_per_batch:
            self.K = num_targets # for each batch define self.K = batch's num_targets.


            negative_sample_indices = torch.tensor([list(torch.arange(start = start_idx, end = start_idx +  num_targets, step = 1 )) for _ in range(num_targets)])
            

            # label の num_targets * num_targets を取り出す。
            tmp = label[negative_sample_indices]
            tmp2 = tmp
            tmp4 = torch.ones( (tmp2.size(0),tmp2.size(1) ), device=torch.device(device)  )
            tmp3 = torch.zeros( num_targets , self.Kmax, tmp2.size(2) , device=torch.device(device) )
            tmp5 = torch.zeros( num_targets , self.Kmax, device=torch.device(device)  )
            tmp3[:,:tmp2.size(1),: ] = tmp2[:,:,:]
            tmp5[:,:tmp2.size(1) ] = tmp4[:,:]

            # cat していくと結果的に (N,Kmax)
            negative_samples = torch.cat( [ negative_samples, tmp3 ], dim = 0 )
            num_targets_mask = torch.cat( [ num_targets_mask, tmp5 ], dim = 0 )
            start_idx += num_targets

            # 各 N についての num_targets を num_neg_sim_sum に格納。 ( N )
            tmp = [ num_targets for _ in range( num_targets )  ]
            tmp = torch.tensor( tmp, device = num_neg_sim_sum.device )
            num_neg_sim_sum = torch.cat( [num_neg_sim_sum, tmp] )
            
        return negative_samples, num_targets_mask, num_neg_sim_sum
