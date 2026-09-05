# SPICA class-conditional alignment protocol

Status: implemented; new runs are separate from the historical frozen-prompt artifacts.

## Phase A gate

The controlled FP3 reference and the alignment arms use the following explicit
protocol.

| Item | Historical FP3 reference | New alignment campaign |
|---|---|---|
| Dataset | Sketchy 104/21 | Same source manifests |
| Model | ViT-B-32-quickgelu, OpenAI weights | Same |
| Query/gallery | Separate visual sketch/photo prompts; raw sketch is the only query input | Same |
| Train/test classes | Pseudo-unseen split, seed 3407, 84/20 classes | Same |
| Selection | Full pseudo-unseen mAP, prefix-positive AP denominator | Same |
| Official unseen | Never used for selection | Never used for selection; diagnostic only |
| Backbone | Frozen CLIP visual and text towers | Frozen CLIP visual and text towers |
| Trainable state | Visual prompts and FP3 soft text context | Same |
| Historical positives | One sampled positive photo | Four sampled positive photos |
| New batches | Ordinary shuffled batches in the historical run | 16 classes × 2 sketches/class = 32 rows |
| Alignment target | None | Positive photos from the current pseudo-train batch only; target moments detached |
| Resume | Historical trainer validates checkpoint role, campaign, split, treatment, optimizer groups, RNG, and checkpoint hashes | Alignment runs are explicitly from scratch; no resume is silently accepted |

The integer IDs in the train and test class maps are reused (`0..20` in the
separate test map), but the normalized semantic names have zero overlap. The
observed audit was therefore an ID-map reuse, not class leakage. The pseudo
split has no class overlap and no path intersection between its train and
validation portions.

The historical FP3 reduction is:

```text
rank = mean(softplus(margin - cosine(q, p+) + cosine(q, p-)))
cls  = cross_entropy(cosine(q, soft_text_bank) / tau_cls, class_id)
total = lambda_rank * rank + lambda_cls * cls
```

`cross_entropy` uses its standard mean reduction. The new control preserves
this objective, changes only the predeclared matched sampler and four-positive
rank reduction (`mean` over the four positive scores), and is the fair control
for the alignment arms.

## Objective

For class `y`, let `a_y` be a frozen hard CLIP text embedding and let `q_s` and
`p_m` be normalized prompted sketch and positive-photo embeddings. The default
arm computes

```text
u_s = Log_{a_y}(q_s)
 u_p = Log_{a_y}(p_m)
mu_s, Sigma_s = moments({u_s})
mu_p, Sigma_p = moments({u_p})
L_align = lambda_mu * ||mu_s - stopgrad(mu_p)||²_2
         + lambda_S * ||Sigma_s - stopgrad(Sigma_p)||²_F
```

Moments are reduced per matched class group and then averaged across usable
classes. The group contains at least two sketches; the photo target contains
`2 × 4 = 8` positive photos. The photo forward pass still receives gradients
from ranking, but its moments are a stop-gradient target for alignment. Text
anchors are also detached. `photo_mean` is a predeclared no-text-anchor control.
`chordal` is a predeclared no-Log-map control.

Alignment is loss-side only. Text and photos are not inputs to the predictor at
inference, and the gallery is encoded only when constructing the ordinary
retrieval gallery. Validation/test examples never form an alignment target.

## Predeclared arms

- `alignment_control`: matched sampler, no alignment.
- `alignment_mean_text_log`: text anchor + Log map + mean only.
- `alignment_cov_text_log`: text anchor + Log map + covariance only.
- `alignment_full_text_log`: text anchor + Log map + mean + covariance.
- `alignment_full_chordal`: mean + covariance without the Log map.
- `alignment_full_photo_anchor`: Log map + mean + covariance with a detached photo-mean anchor instead of text.

## Literature comparison

| Status | Work | Confirmed result | Relation to this campaign |
|---|---|---|---|
| Literature-confirmed | Wu et al., *Distribution Aligned Feature Clustering for Zero-Shot SBIR*, arXiv:2301.06685 | Aligns sketch and image features with a common Gaussian distribution and uses gallery clustering for retrieval | Distribution alignment and gallery multi-photo use are prior art; this code does not claim them as new |
| Literature-confirmed | Liu et al., *Symmetrical Bidirectional Knowledge Alignment for Zero-Shot SBIR*, arXiv:2312.10320 | Uses a Gaussian prior/KL-style inter-modality alignment and a one-to-many gallery matching strategy | Confirms Gaussian/domain alignment and multi-photo aggregation in ZS-SBIR; it is not this class-conditional text-anchored tangent-moment loss |
| Literature-confirmed | Sun & Saenko, *Deep CORAL*, arXiv:1607.01719 | Aligns second-order activation statistics across domains | Confirms covariance alignment as a general domain-adaptation pattern |
| Literature-confirmed | Singha et al., *SpLIP*, arXiv:2407.04207 | Uses frozen CLIP, multimodal prompt learning, text-image classification, and a text-derived adaptive margin for SBIR | Confirms CLIP/text supervision and prompt learning in this problem family |
| Proposed difference | This campaign | Class-conditional sketch/photo first and second moments in a CLIP-sphere tangent plane, with hard class-text anchors and detached positive-photo targets, evaluated under a sketch-only predictor contract | The combination and exact implementation are the experiment's proposed difference |
| Unverified novelty | — | No direct literature match was established for the exact text-anchor + spherical Log map + class-conditional covariance objective in this repository's search | Do not call this combination novel until a broader bibliographic/code search verifies it |

Sources: [2301.06685](https://arxiv.org/abs/2301.06685),
[2312.10320](https://arxiv.org/abs/2312.10320),
[1607.01719](https://arxiv.org/abs/1607.01719), and
[2407.04207](https://arxiv.org/abs/2407.04207).
