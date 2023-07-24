idx = 4

image = new_dataset[idx]["image"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(new_dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)

model.eval()

# forward pass
with torch.no_grad():
  outputs = model(**inputs, multimask_output=False)


# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

fig, axs = plt.subplots(ncols=3, figsize=(15, 7))
axs[1].imshow(image)
axs[1].set_title("Input")
axs[0].imshow(ground_truth_mask)
axs[0].set_title('"Ground truth"')
axs[2].imshow(medsam_seg)
axs[2].set_title("Prediction")

for ax in axs:
    ax.set_axis_off()

fig.show()


original_model = SamModel.from_pretrained("facebook/sam-vit-base")
original_model.to(device)
print()

original_model.eval()

# forward pass
with torch.no_grad():
  outputs = original_model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(medsam_seg)
ax.set_title("No fine-tuning prediction")
ax.set_axis_off()
fig.show()