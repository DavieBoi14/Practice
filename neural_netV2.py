import math
import numpy
import random

# A = np.array([
#     [1, 2],
#     [3, 4]
# ])

# B = np.array([
#     [5, 6],
#     [7, 8]
# ])

# result = A @ B



class Layer:

    def __init__(self, inputs, num_outputs, weight_varience):
        self.weight_varience = weight_varience
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.weights = []
        self.biases = []
        self.output = []
    

    def create_weights(self):
        weights = []

        for i in range(self.num_outputs):

            row = []
            for j in range(len(self.inputs)):
                row.append(random.randint(self.weight_varience * -1, self.weight_varience))
            
            weights.append(row)
            row = []

        return weights

    def create_biases(self):

        biases = []

        for i in range(self.num_outputs):
            biases.append(random.randint(self.weight_varience * -1, self.weight_varience))
        
        return biases

    def calculate_outputs(self):

        weights = self.create_weights()
        biases = self.create_biases()

        self.weights = weights
        self.biases = biases

      
        W = numpy.array(weights)
        I = numpy.array(self.inputs)
        B = numpy.array(biases)
        
        outputs = W @ I + B 

        self.output = outputs

        return outputs
    
    def print_weights(self):
        print(self.weights)
    
    def print_biases(self):
        print(self.biases)

    def print_outputs(self):
        print(self.output)
    

class Network:

    def __init__(self, structure, inputs, weight_varience, desired_outputs):
        self.desired_outputs = desired_outputs
        self.inputs = inputs
        self.weight_varience = weight_varience
        self.structure = structure
        self.weights = []
        self.biases = []
        self.outputs = []
        self.error = None

    def calculate_error(self):

        total_error = 0

        for output, desired_output in zip(self.outputs, self.desired_outputs):
            error = (output - desired_output)**2
            total_error += error

        self.error = total_error
        return total_error
            

    def calculate_outputs(self):

        first_time = True
        for layer in self.structure:

            if first_time:
                Layer_object = Layer(self.inputs, layer, self.weight_varience)
                outputs = Layer_object.calculate_outputs()
                first_time = False
            else:
                Layer_object = Layer(outputs, layer, self.weight_varience) 
                outputs = Layer_object.calculate_outputs()
            
            self.weights.append(Layer_object.weights)
            self.biases.append(Layer_object.biases)

            # Layer_object.print_weights()
            # Layer_object.print_biases()
            # Layer_object.print_outputs()
        
        self.outputs = outputs
        return outputs





Network_Test = Network([2,3], [1,1,1], 2, [5,5,5])     # layer structure, inputs, weight varience, desired output 

Network_Test.calculate_outputs()

Network_Test.calculate_error()



# Layer_test = Layer([1,5,3], 2, 5)
# Layer_test.calculate_outputs()