How to interpret the overlay image in the demonstration video:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The overlay image (bottom right quadrant) shows:
	The original X-ray image in grayscale
	The model's prediction overlaid in red
Here's how to interpret it:
1. Original X-ray (Grayscale):
	Darker areas represent denser tissues (like bones)
	Lighter areas represent less dense tissues (like lungs)
	The rib cage appears as white curved lines
	The lungs appear as darker regions between the ribs
2. Model's Prediction (Red Overlay):
	The red color indicates where the model thinks the lungs are
	More intense red = higher confidence in the prediction
	The red overlay is semi-transparent (70% opacity) so you can still see the original X-ray underneath
3. What to Look For:
	Good Segmentation: The red overlay should closely match the actual lung boundaries
	Over-segmentation: If red areas extend beyond the actual lungs
	Under-segmentation: If parts of the lungs are not covered by the red overlay
	Edge Accuracy: How well the red boundaries align with the actual lung edges
4. Comparing with Ground Truth:
	You can compare the overlay with the ground truth mask (top right quadrant)
	The ground truth shows the actual lung boundaries as annotated by medical experts
	Any differences between the red overlay and the ground truth indicate areas where the model needs improvement
