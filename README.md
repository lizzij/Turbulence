# Combine Equivariant Models

## Rot-UM
1. test for equivariance: check UM and Rot eq
2. train on untransformed, test on UM, Rot, UM + Rot
3. train on random UM + Rot, test on UM + Rot (same dataset)
  - how to generate UM + Rot dataset: add UM (diff vec - selected from a circle) for u and v), then Rot 
  - rmse only on the 64 * 64 (check vs all)

## Rot-UM-Mag

## Rot-Um-Scale
