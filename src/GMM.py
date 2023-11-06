"""
Authors: Frederik Hartmann and Xavier Beltran
Date: 25-10-2023
"""
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from util import Util


class GaussianMixtureModel:
    # init all the variables
    def __init__(self, k, data, maxIterations=500):
        self.maxIterations = maxIterations
        self.data = data
        self.k = k  # number of Clusters

        self.dimensions = data.shape[1]
        self.numberOfPixel = data.shape[0]

        self.membershipWeights = np.full((self.k, self.numberOfPixel), 1 / self.k)
        self.mixtureWeights = np.full(self.k, 1 / self.k)

        self.covariance = np.zeros((self.k, self.dimensions, self.dimensions))
        self.mean = np.zeros((self.k, self.dimensions))

        self.prevLog = 0
        self.completedIterations = 0
        self.tolerance = 1e-3

        # division by zero
        self.epsilon = 1e-6

    def run(self):
        # main loop to run the code
        while (not self.isConverged()):
            self.expectationStep()
            self.maximizationStep()
            self.completedIterations += 1
            print(f" Iteration {self.completedIterations} of {self.maxIterations}")  # , end='\r')
        clusterAssignments = np.argmax(self.membershipWeights, axis=0)
        return clusterAssignments

    def expectationStep(self):
        self.computeMembershipWeights()

    def maximizationStep(self):
        self.updateMixtureWeights()
        self.updateMeans()
        self.updateCovariance()

    def initialization(self, initialization_type):
        # initialize means randomly or with KMeans
        if initialization_type == "Random":
            np.random.seed(11)
            self.mean = np.random.uniform(0, 80, (self.k, self.dimensions))
            self.covariance = np.array([self.randDiagCovarianceMatrix() for _ in range(self.k)])

        elif initialization_type == "KMeans":
            kmeans = KMeans(n_clusters=self.k, n_init="auto").fit(self.data)
            self.mean = kmeans.cluster_centers_
            self.covariance = np.array([self.randDiagCovarianceMatrix() for _ in range(self.k)])
        else:
            raise ("allowed initialization types are Random and KMeans")
        pass

    def randDiagCovarianceMatrix(self):
        # return a random diagonal matrix
        np.random.seed(42)
        diagVector = np.random.uniform(5, 10, self.dimensions)
        return np.diag(diagVector)

    def computeMembershipWeights(self):
        # compute membership weights using the gaussian density function
        weightedProbabilities = []
        for cluster in range(self.k):
            probability = self.gaussianDensityFunction(cluster)

            weightedProbability = probability * self.mixtureWeights[cluster]
            weightedProbabilities.append(weightedProbability)

        weightedProbabilities = np.array(weightedProbabilities)
        sumOfWeightedProbabilities = np.sum(weightedProbabilities, axis=0)
        sumOfWeightedProbabilities = self.avoidDivisionByZero(sumOfWeightedProbabilities)
        for cluster in range(self.k):
            self.membershipWeights[cluster] = weightedProbabilities[cluster] / sumOfWeightedProbabilities

    def gaussianDensityFunction(self, cluster):
        # compute probability using a multimodal gaussian probality
        if self.isSingularMatrix(self.covariance[cluster]):
            self.covariance[cluster].flat[:: self.dimensions + 1] += self.epsilon  # add epsilon to main diagonal

        det_cov = np.linalg.det(self.covariance[cluster])
        inv_cov = np.linalg.inv(self.covariance[cluster])

        diff = self.data - self.mean[cluster]
        exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        part1 = 1 / (((2 * np.pi) * (self.dimensions / 2)) * (det_cov * (1 / 2)))
        return part1 * np.exp(exponent)

    @staticmethod
    def isSingularMatrix(matrix):
        # checls if the matrix is a singular matrix
        if np.linalg.cond(matrix) < 1 / sys.float_info.epsilon:
            return False
        else:
            return True

    def updateMixtureWeights(self):
        # updates the mixture weights (alpha)
        for cluster in range(self.k):
            self.mixtureWeights[cluster] = np.sum(self.membershipWeights[cluster], axis=0) / self.numberOfPixel

    def updateMeans(self):
        # update the means of the clusters
        divisor = (np.sum(self.membershipWeights.T, axis=0) + 10 * np.finfo("float64").eps)
        self.mean = self.membershipWeights @ self.data / divisor[:, np.newaxis]

    def updateCovariance(self):
        # update the covariance matrix of the clustsers
        for cluster in range(self.k):
            part1 = 1 / (np.sum(self.membershipWeights[cluster], axis=0) + 10 * np.finfo("float64").eps)
            diff = self.data - self.mean[cluster]
            self.covariance[cluster] = part1 * np.dot(self.membershipWeights[cluster] * diff.T, diff)

    def isConverged(self):
        # checks if the algorithm converged by comparing the new log with the previous log
        newLog = self.logLikelyhood()

        if self.completedIterations >= self.maxIterations:
            print(f"Maximum iterations ({self.maxIterations}) reached")
            return True
        elif np.abs(newLog - self.prevLog) < self.tolerance:
            print("converged")
            return True
        else:
            self.prevLog = newLog
            return False

    def logLikelyhood(self):
        # computes the log likelyhood of a multimodal gaussian
        weightedProbabilities = []
        for cluster in range(self.k):
            probability = self.gaussianDensityFunction(cluster)
            weightedProbability = probability * self.mixtureWeights[cluster]
            weightedProbabilities.append(weightedProbability)
        weightedProbabilities = np.array(weightedProbabilities)
        sumOfWeightedProbabilities = np.sum(weightedProbabilities, axis=0)
        sumOfWeightedProbabilities = self.avoidDivisionByZero(sumOfWeightedProbabilities)
        logOfSum = np.log(sumOfWeightedProbabilities)
        return np.sum(logOfSum, axis=0)

    def avoidDivisionByZero(self, array):
        # checks if the array contains zeros and adds a small number if its true
        if array.any():
            array[array == 0] += self.epsilon
        return array


if __name__ == "__main__":
    testFolder = "/home/frederik/Dokumente/Master/Semester_3/misa/lab2/P2_Data/"
    case = 1
    util = Util()

    GT, affine = util.readNiftiImage(testFolder + f"{case}/LabelsForTesting.nii")
    T1, _ = util.readNiftiImage(testFolder + f"{case}/T1.nii")
    T2, _ = util.readNiftiImage(testFolder + f"{case}/T1.nii")

    # Flatten and concatenate the data
    both_modalities = np.stack([T1[GT > 0].flatten(), T2[GT > 0].flatten()], axis=1)
    one_modality = T1[GT > 0].reshape(-1, 1)

    n_classes = 3
    # Start GMM Algorithm
    GMM = GaussianMixtureModel(k=n_classes, data=both_modalities)
    GMM.initialization(initialization_type="Random")
    cluster_assignments = GMM.run()

    # Reconstruct Image
    reverted_image_ = util.reconstruct_image(cluster_assignments, GT)
    reverted_image = util.fitLabelToGT(reverted_image_.astype(np.uint8), GT.astype(np.uint8), n_classes)

    slice_index = 20  # Specify the slice index you want to display

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots

    # Plot reverted_image on the first subplot
    axes[0].imshow(reverted_image[:, :, slice_index], cmap='gray')
    axes[0].set_title('Reverted Image')

    # Plot GT on the second subplot
    axes[1].imshow(GT[:, :, slice_index], cmap='gray')
    axes[1].set_title('Ground Truth')

    plt.show()

    print(f"\n-----------------RESULTS------------------")
    print(f" WM DSC: {util.dice_coefficient((reverted_image == 1).astype(np.uint8), (GT == 1).astype(np.uint8))}")
    print(f" GM DSC: {util.dice_coefficient((reverted_image == 2).astype(np.uint8), (GT == 2).astype(np.uint8))}")
    print(f" CSF DSC: {util.dice_coefficient((reverted_image == 3).astype(np.uint8), (GT == 3).astype(np.uint8))}")