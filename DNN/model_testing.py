import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GroupNormalization, BatchNormalization
import models
import numpy as np


# Assuming your get_MobileNet function is already defined
# (it should include the replace_batchnorm_with_groupnorm logic)

# Create the original model with BatchNormalization
bn_model = models.get_MobileNet(K=1000, IsGN=False)

# Create the modified model with GroupNormalization
gn_model = models.get_MobileNet(K=1000, IsGN=True, group_sz=2)

print("Layer-by-Layer Comparison of BatchNormalization vs. GroupNormalization Models\n")
print(f"{'Layer Name':<40} | {'BN Layer Type':<20} | {'GN Layer Type':<20}")
print("-" * 85)

# The models have the same number of layers and a similar structure.
# We can iterate through them and compare corresponding layers.
for i in range(len(bn_model.layers)):
    bn_layer = bn_model.layers[i]
    gn_layer = gn_model.layers[i]
    
    # Check if the layers are of different types, which is what we expect
    # for normalization layers.
    if type(bn_layer) != type(gn_layer):
        # We've found a difference! Let's check if it's a normalization layer.
        if isinstance(bn_layer, BatchNormalization) and isinstance(gn_layer, GroupNormalization):
            print(f"{bn_layer.name:<40} | {type(bn_layer).__name__:<20} | {type(gn_layer).__name__:<20}  ✅")
        else:
            # If there's an unexpected difference, print it as an error.
            print(f"{bn_layer.name:<40} | {type(bn_layer).__name__:<20} | {type(gn_layer).__name__:<20}  ❌ UNEXPECTED DIFFERENCE")
    else:
        # If the layers are the same type, just print them for context.
        print(f"{bn_layer.name:<40} | {type(bn_layer).__name__:<20} | {type(gn_layer).__name__:<20}")

print("\n--- Summary of Differences ---")

# Count the number of normalization layers replaced
bn_norm_count = sum(1 for layer in bn_model.layers if isinstance(layer, BatchNormalization))
gn_norm_count = sum(1 for layer in gn_model.layers if isinstance(layer, GroupNormalization))

print(f"Original model had {bn_norm_count} BatchNormalization layers.")
print(f"Modified model has {gn_norm_count} GroupNormalization layers.")

# Compare total parameter counts
bn_params = bn_model.count_params()
gn_params = gn_model.count_params()

print(f"\nTotal parameters in original (BN) model: {bn_params}")
print(f"Total parameters in modified (GN) model: {gn_params}")
print(f"Difference in parameters: {bn_params - gn_params}")

print("Verifying weights for non-normalization layers:")
print("-" * 50)

for i in range(len(bn_model.layers)):
    bn_layer = bn_model.layers[i]
    gn_layer = gn_model.layers[i]

    # Skip normalization layers as their weights are intentionally different
    if not isinstance(bn_layer, (BatchNormalization, GroupNormalization)):
        bn_weights = bn_layer.get_weights()
        gn_weights = gn_layer.get_weights()
        
        # Check if weights exist for both layers
        if bn_weights and gn_weights:
            # Use numpy.array_equal to check for element-wise equality
            weights_match = all(np.array_equal(w1, w2) for w1, w2 in zip(bn_weights, gn_weights))
            if weights_match:
                print(f"✅ Weights match for layer: {bn_layer.name}")
            else:
                print(f"❌ Weights DO NOT match for layer: {bn_layer.name}")
        else:
            # Handle layers with no weights (e.g., Activation layers)
            print(f"--- No weights to compare for layer: {bn_layer.name}")

random_input = tf.random.normal(shape=(1, 224, 224, 3))

# Get the output from both models
bn_output = bn_model(random_input)
gn_output = gn_model(random_input)

# Calculate the mean squared difference between the outputs
difference = np.mean(np.square(bn_output - gn_output))

print("\nVerifying layer connections by comparing model outputs:")
print("-" * 50)
print(f"Mean Squared Difference between BN and GN model outputs: {difference:.8f}")

# A small, non-zero difference is expected and a good sign.
# A difference of 0 would mean the GroupNormalization layer isn't doing anything,
# while a very large difference might indicate a structural issue.
# The expected difference is small because most of the weights are the same,
# but the normalization is different.