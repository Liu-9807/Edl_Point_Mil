from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

# ===========================================================================
#  Helper Functions & Sub-Modules (From evmil.py / deepmil.py)
# ===========================================================================

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def elu_evidence(y):
    return F.elu(y) + 1

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    """DSMIL Bag Classifier adapted for MMDetection flow."""
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0):
        super(BClassifier, self).__init__()
        self.output_class = output_class
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)

    def forward(self, feats, c): 
        # feats: [N, K], c: [N, C] (instance scores)
        device = feats.device
        V = self.v(feats) # [N, V]
        Q = self.q(feats) # [N, Q]
        
        # Critical Instance Selection
        _, m_indices = torch.sort(c, 0, descending=True) # [N, C]
        # Select critical instances for each class
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # [C, K]
        q_max = self.q(m_feats) # [C, Q]
        
        # Attention Calculation
        # A: [N, C]
        A = torch.mm(Q, q_max.transpose(0, 1)) 
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) 
        
        # Aggregation
        B = torch.mm(A.transpose(0, 1), V) # [C, V]
        B = B.view(1, B.shape[0], B.shape[1]) # [1, C, V]
        
        # Classification
        C = self.fcc(B) # [1, C, 1]
        C = C.view(1, -1) # [1, C]
        return C, A

class ScoringNet(nn.Module):
    """ABMIL Attention Scoring Net."""
    def __init__(self, dim_in, dim_hid):
        super(ScoringNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.Tanh(),
            nn.Linear(dim_hid, 1)
        )
    def forward(self, x):
        # x: [N, C] -> a: [N, 1]
        a = self.net(x)
        return a

class GatedScoringNet(nn.Module):
    """ABMIL Gated Attention Scoring Net."""
    def __init__(self, dim_in, dim_hid):
        super(GatedScoringNet, self).__init__()
        self.V = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.Sigmoid())
        self.w = nn.Linear(dim_hid, 1)
    def forward(self, x):
        A = self.w(self.V(x) * self.U(x))
        return A

# ===========================================================================
#  Main EDLHead Class
# ===========================================================================

@MODELS.register_module()
class EDLHead(BaseModule):
    """
    Unified MIL Head supporting DeepMIL, ABMIL, and DSMIL with Evidential Output.
    Encapsulates logic from DeepMIL.py and Evmil.py.
    """
    def __init__(self,
                 num_classes=2,
                 in_channels=256,
                 hidden_channels=1024,
                 mil_type='ab',     # 'deep', 'ab', 'ds'
                 pooling_type='gated', # For DeepMIL: 'max'/'mean'; For ABMIL: 'attention'/'gated'
                 ins_enhance=False,
                 use_frozen_feat_in_eins=True,
                 loss_edl=dict(type='EDLLoss', loss_type='mse', loss_weight=0.5),
                 loss_aux=dict(type='EDLLoss', loss_type='mse', loss_weight=0.5),
                 edl_evidence_func='relu',
                 init_cfg=None,
                 **kwargs):
        super(EDLHead, self).__init__(init_cfg)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.mil_type = mil_type
        self.pooling_type = pooling_type
        self.ins_enhance = ins_enhance
        self.use_frozen_feat_in_eins = use_frozen_feat_in_eins

        # 1. Feature Projector (Feature Extractor in DSMIL terms)
        # Assuming simple linear projection as default in reference code
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 2. MIL Specific Architectures
        if self.mil_type == 'deep':
            # DeepMIL (Max/Mean)
            assert pooling_type in ['max', 'mean']
            self.classifier = nn.Linear(hidden_channels, num_classes)
            
        elif self.mil_type == 'ab':
            # ABMIL (Attention)
            if pooling_type == 'attention':
                self.att_net = ScoringNet(hidden_channels, hidden_channels)
            elif pooling_type == 'gated':
                self.att_net = GatedScoringNet(hidden_channels, hidden_channels)
            else:
                raise ValueError(f"Unknown pooling type {pooling_type} for ABMIL")
            self.classifier = nn.Linear(hidden_channels, num_classes)

        elif self.mil_type == 'ds':
            # DSMIL (Dual-Stream)
            self.ds_i_classifier = FCLayer(hidden_channels, num_classes)
            self.ds_b_classifier = BClassifier(hidden_channels, hidden_channels, num_classes)
        else:
            raise NotImplementedError(f"MIL type {mil_type} not implemented")

        # 3. Evidence Activation Function
        if edl_evidence_func == 'relu':
            self.evidence_func = relu_evidence
        elif edl_evidence_func == 'exp':
            self.evidence_func = exp_evidence
        elif edl_evidence_func == 'softplus':
            self.evidence_func = softplus_evidence
        elif edl_evidence_func == 'elu':
            self.evidence_func = elu_evidence
        else:
            raise ValueError("Unknown evidence function")

        # 4. Instance Enhancement (Residual)
        if self.ins_enhance:
            self.ins_enhance_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, num_classes),
                nn.Tanh() # Output scale in [-1, 1]
            )
            
            # If we need a classifier for initial instance prediction in Deep/AB modes
            # (DSMIL already has i_classifier)
            if self.mil_type != 'ds':
                self.ins_classifier_aux = nn.Linear(hidden_channels, num_classes)

        # 5. Losses
        self.loss_edl = MODELS.build(loss_edl)
        self.loss_aux = MODELS.build(loss_aux)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _forward_deep(self, bag_feats):
        """DeepMIL forward: mean/max pooling -> classifier."""
        # bag_feats: [N, C]
        if self.pooling_type == 'max':
            bag_emb, _ = torch.max(bag_feats, dim=0, keepdim=True) # [1, C]
        else: # mean
            bag_emb = torch.mean(bag_feats, dim=0, keepdim=True)   # [1, C]
        
        bag_logits = self.classifier(bag_emb) # [1, num_classes]
        
        # For DeepMIL, instance logits usually come from the same classifier or auxiliary
        if self.ins_enhance:
            ins_logits = self.ins_classifier_aux(bag_feats) # [N, num_classes]
        else:
            ins_logits = self.classifier(bag_feats) # [N, num_classes] (weak supervision assumption)

        return bag_logits, ins_logits

    def _forward_ab(self, bag_feats):
        """ABMIL forward: attention -> weighted sum -> classifier."""
        # bag_feats: [N, C]
        # 1. Calculate Attention
        att_scores = self.att_net(bag_feats) # [N, 1]
        att_weights = F.softmax(att_scores, dim=0) # [N, 1]
        
        # 2. Weighted Sum
        bag_emb = torch.sum(att_weights * bag_feats, dim=0, keepdim=True) # [1, C]
        
        # 3. Classify
        bag_logits = self.classifier(bag_emb) # [1, num_classes]
        
        # Initial instance logits
        if self.ins_enhance:
             ins_logits = self.ins_classifier_aux(bag_feats)
        else:
             # Just use attention scores as proxy logic, or same classifier
             ins_logits = self.classifier(bag_feats) 
             
        return bag_logits, ins_logits

    def _forward_ds(self, bag_feats):
        """DSMIL forward: dual stream."""
        # 1. Instance Stream
        # feats: [N, C], logits: [N, num_classes]
        feats, ins_logits = self.ds_i_classifier(bag_feats) 
        
        # 2. Bag Stream (guided by critical instance)
        # bag_logits_part: [1, num_classes], A: Attention
        bag_logits_part, _ = self.ds_b_classifier(feats, ins_logits)
        
        # 3. Max pooling on instance scores
        max_ins_logits, _ = torch.max(ins_logits, dim=0, keepdim=True) # [1, num_classes]
        
        # 4. Fusion
        bag_logits = 0.5 * (bag_logits_part + max_ins_logits)
        
        return bag_logits, ins_logits

    def forward(self, x, rois):
        """
        Args:
            x (Tensor): [Total_N, C_in, H, W] or [Total_N, C_in]
            rois (Tensor): [Total_N, 5], first col is batch_ind
        Returns:
            bag_alpha (Tensor): [B, Num_Classes]
            ins_output (Tensor/Tuple): Depends on enhancement
        """
        # 1. Pre-process input
        if x.dim() == 4:
            x = self.avg_pool(x).flatten(1)
        
        # Shared projection
        x_proj = self.feat_proj(x) # [Total_N, Hidden]
        
        batch_size = int(rois[:, 0].max().item() + 1)
        
        bag_alpha_list = []
        ins_output_list = [] # Store logic specific to loss requirement

        # 2. Process per Bag (Image)
        for i in range(batch_size):
            # Extract instances for current image
            inds = torch.where(rois[:, 0] == i)[0]
            if len(inds) == 0:
                # Handle empty bag case (sanity check)
                # Create dummy gradient-capable tensor
                dummy_alpha = torch.ones(1, self.num_classes, device=x.device, requires_grad=True)
                bag_alpha_list.append(dummy_alpha)
                # Should append empty tensor for instances to match logic
                continue
                
            bag_feats = x_proj[inds] # [N_i, Hidden]
            
            # --- Generic MIL Dispatch ---
            if self.mil_type == 'deep':
                bag_logits, ins_logits_init = self._forward_deep(bag_feats)
            elif self.mil_type == 'ab':
                bag_logits, ins_logits_init = self._forward_ab(bag_feats)
            elif self.mil_type == 'ds':
                bag_logits, ins_logits_init = self._forward_ds(bag_feats)
            
            # --- EDL Transformation (Bag) ---
            bag_evidence = self.evidence_func(bag_logits) # [1, Num_Classes]
            bag_alpha = bag_evidence + 1
            bag_alpha_list.append(bag_alpha)
            
            # --- Instance Enhancement Logic (MIREL) ---
            if self.ins_enhance:
                # 1. Get initial evidence (freeze or not treated inside backprop graph depends on architecture)
                init_ins_evidence = self.evidence_func(ins_logits_init).detach().clone() # [N_i, Num_Classes]
                
                # 2. Calculate Scale
                # DSMIL specifically uses un-projected feats in some ref code, but here we aligned to x_proj
                # Use clone to avoid modifying graph if needed, but linear layers need grad
                if self.use_frozen_feat_in_eins:
                    feat_for_scale = bag_feats.detach().clone()
                else:
                    feat_for_scale = bag_feats
                    
                scale = self.ins_enhance_layer(feat_for_scale) # [N_i, Num_Classes]
                
                # 3. Apply Residual Multiplier
                # Formula from evmil.py: enhanced = init ** (1 + scale)
                enhanced_ins_evidence = init_ins_evidence ** (1 + scale)
                
                enhanced_ins_alpha = enhanced_ins_evidence + 1
                init_ins_alpha = init_ins_evidence + 1
                
                # Store tuple for loss usage
                # We need to stack them later, so store as tensor is better if possible.
                # However, lengths differ. So we store list of tensors.
                ins_output_list.append((init_ins_alpha, enhanced_ins_alpha))
            else:
                # No enhancement, just store logits for cross-entropy aux loss
                # OR store alpha if aux loss is EDL.
                # Assuming Aux is CrossEntropy per default config in MMDetection Heads
                ins_output_list.append(ins_logits_init) 

        # 3. Collate Results
        final_bag_alpha = torch.cat(bag_alpha_list, dim=0) # [B, Num_Classes]
        
        # Instance outputs are ragged (different N per bag), cannot stack simply if we want to preserve structure
        # But MILRoIHead expects a flat tensor for 'ins_score' usually to match flat 'ins_labels'
        # Let's flatten instance outputs
        if self.ins_enhance:
            # Flatten tuples: (Total_N, C), (Total_N, C)
            init_alls = torch.cat([x[0] for x in ins_output_list], dim=0)
            enh_alls = torch.cat([x[1] for x in ins_output_list], dim=0)
            final_ins_output = (init_alls, enh_alls)
        else:
            final_ins_output = torch.cat(ins_output_list, dim=0) # [Total_N, Num_Classes]

        # [新增] 仅在此时暂存数据用于 Hook 可视化
        if hasattr(self, 'save_debug_info') and self.save_debug_info:
            self._last_debug_data = {
                'bag_alpha': final_bag_alpha.detach().cpu(),
                'ins_output': final_ins_output[1].detach().cpu(),  # use enhanced alpha 
                'rois': rois.detach().cpu(), # 这里的 rois 第一列是 batch_idx，后四列是 coords
            }

        return final_bag_alpha, final_ins_output

    def loss(self, cls_score, bag_label, ins_labels=None, epoch_num=0):
        """
        Args:
            cls_score: (bag_alpha, ins_output) from forward
            bag_label: [B]
            ins_labels: [Total_N]
        """
        bag_alpha, ins_output = cls_score
        losses = {}
        
        # 1. Bag Level EDL Loss
        if self.loss_edl.loss_weight > 0:
            bag_label_onehot = F.one_hot(bag_label, num_classes=self.num_classes).float()
            # edl_loss.py expects inputs: (alpha, target, epoch_num, num_classes...)
            loss_bg = self.loss_edl(
                bag_alpha, 
                bag_label_onehot, 
                epoch_num=epoch_num, 
            )
            losses['loss_edl_bag'] = loss_bg

        # 2. Instance Level Aux Loss
        if self.loss_aux.loss_weight > 0 and ins_labels is not None:
            if self.ins_enhance:
                # ins_output is (init_alpha, enhanced_alpha)
                # Usually MIREL applies loss to both or just enhanced?
                # evmil.py usually computes loss on both but we simplify to Enhanced logic here
                # unless loss_aux supports tuples.
                # Assuming loss_aux is CE, it expects Logits. but we have Alpha here.
                # We should use EDL loss for instance part if we have Alpha.
                # Check config: If ins_enhance is True, aux loss MUST be EDL compatible or we convert back?
                # No, standard MIREL uses EDL loss for instances too.
                
                init_alpha, enh_alpha = ins_output
                ins_label_onehot = F.one_hot(ins_labels, num_classes=self.num_classes).float()
                
                # Apply EDL loss to enhanced instance predictions
                loss_ins_enh = self.loss_edl(
                    enh_alpha,
                    ins_label_onehot,
                    epoch_num=epoch_num,
                )
                losses['loss_edl_ins'] = loss_ins_enh * self.loss_aux.loss_weight
                
            else:
                # Standard Mode: ins_output is Logits
                # Use standard CrossEntropy (or whatever loss_aux is)
                # loss_aux call in mmdet usually handles (pred, label)
                # label needs to be long for CE
                loss_ins = self.loss_aux(ins_output, ins_labels)
                losses['loss_aux_ins'] = loss_ins

        return losses