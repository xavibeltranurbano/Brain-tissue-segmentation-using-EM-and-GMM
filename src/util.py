"""
Authors: Frederik Hartmann and Xavier Beltran
Date: 25-10-2023
"""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import seaborn as sns


class Util:
    def _init_(self):
        pass

    def reconstruct_image(self, cluster_assignments, GT):
        # Reconstruct segmentation to the original shape
        reconstruct_image = np.zeros(GT.shape)
        counter = 0
        for i in range(GT.shape[0]):
            for j in range(GT.shape[1]):
                for k in range(GT.shape[2]):
                    if GT[i, j, k] > 0:
                        reconstruct_image[i, j, k] = cluster_assignments[counter] + 1
                        counter += 1
        return reconstruct_image

    def readNiftiImage(self, filePath):
        #Read Nifti image
        try:
            niftiImage = nib.load(filePath).get_fdata()
            return niftiImage, nib.load(filePath).affine
        except Exception as e:
            print(f"Error reading NIFTI image from {filePath}: {str(e)}")

    def dice_coefficient(self, segmentation_mask, GT_mask):
        # Compute the Dice score
        intersection = (segmentation_mask & GT_mask).sum()
        total_area = segmentation_mask.sum() + GT_mask.sum()
        dice = (2.0 * intersection) / total_area
        return dice

    def fitLabelToGT(self, segmentation, gt, n_classes):
        # Re-assign labels using the Dice score
        matchingImage = np.zeros(gt.shape)

        for Segmentationcluster in range(n_classes + 1):
            dices = []
            for GTCluster in range(n_classes + 1):
                segBinary = np.where(segmentation == Segmentationcluster, 1, 0)
                gtBinary = np.where(gt == GTCluster, 1, 0)

                dice = self.dice_coefficient(segBinary, gtBinary)
                dices.append(dice)
            correctCluster = np.argmax(dices)
            matchingImage[segmentation == Segmentationcluster] = correctCluster
        return matchingImage

    def plot_original_images(self, vec_img, case, axes, title, slice=20):
        # Create a subplot with 3 rows and 5 columns

        # Loop through each case and its images, and plot them in the subplots
        for j in range(len(vec_img)):  # 3 images per case
            ax = axes[j, case - 1]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= 2:
                ax.imshow(vec_img[j][:, :, slice], cmap='viridis')
            else:
                ax.imshow(vec_img[j][:, :, slice], cmap='gray')

            if j == 0:
                ax.set_title(f'Case {case}')

            if case == 1:
                ax.set_ylabel(title[j])
    def Display_Boxplot(self, xcol, ycol, title_legend, loc='lower right', path_img='', save_fig=False):
        # Display the boxplot
        fontsize_labels = 16
        fontsize_legend = 14
        fontsize_legend_title = 14
        fontsize_ticks = 14
        sns.despine()
        plt.grid(axis='y')
        plt.xlabel(xcol, fontsize=fontsize_labels)
        plt.ylabel(ycol, fontsize=fontsize_labels)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.legend(title=title_legend, loc=loc, fontsize=fontsize_legend,
                   title_fontsize=fontsize_legend_title)
        if save_fig:
            plt.savefig(path_img)
        plt.show()

    def save_image(self, img, case, modality, init_type, affine):
        # Save the images in Nifti format
        nii_image = nib.Nifti1Image(img, affine=affine)  # Replace affine with the appropriate transformation matrix
        file_path = os.path.join(case, f"{init_type}_{modality}.nii")

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(case)

        # Check for write permissions
        if os.access(os.path.dirname(file_path), os.W_OK):
            try:
                nib.save(nii_image, file_path)
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")
        else:
            raise PermissionError(f"No write access to the directory: {os.path.dirname(file_path)}")