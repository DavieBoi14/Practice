import math
import numpy
import random
import json
import matplotlib.pyplot as plt
   

class Network:

    def __init__(self, structure, weight_varience, learning_rate):
        
        self.weight_varience = weight_varience
        self.structure = structure
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.outputs = []
        
        self.create_weights()
        self.create_biases()

   
    def create_weights(self):
        weights = []

        for i in range(len(self.structure) - 1):

            inputs = self.structure[i]
            outputs = self.structure[i+1]

            # print("inputs")
            # print(inputs)

            # print("outputs")
            # print(outputs)
            
            layer_weights = []

            for j in range(outputs):

                row = []

                for k in range(inputs):
                    row.append(random.uniform(self.weight_varience * -1, self.weight_varience))
            

                layer_weights.append(row)
            
            weights.append(layer_weights)


        self.weights = weights
        return weights

    def create_biases(self):

        biases = []

        for i in range(len(self.structure) - 1):

            num_biases_to_make = self.structure[i + 1]

            layer_biases = []

            for j in range(num_biases_to_make):
                layer_biases.append(random.uniform(self.weight_varience * -1, self.weight_varience))
            
            biases.append(layer_biases)
            
        self.biases = biases
       
        return biases


    def calculate_error(self, final_outputs, desired_outputs):

        total_error = 0

        for output, desired_output in zip(final_outputs, desired_outputs):
            error = (output - desired_output)**2
            total_error += error

        return total_error
            

    
    def apply_RELU(self, outputs):
        
        relu_outputs = []

        for output in outputs:
            if (output < 0):
                output = 0
                relu_outputs.append(output)
            else: 
                relu_outputs.append(output)
                
        
        return relu_outputs


    def calculate_outputs_single_layer(self, weights, biases, inputs, is_final_time):
      
        W = numpy.array(weights)
        I = numpy.array(inputs)
        B = numpy.array(biases)
        
        outputs = (W @ I + B).tolist()
    
        if (not is_final_time):
            outputs_with_Relu = self.apply_RELU(outputs)
            return outputs_with_Relu

        else:
            return outputs

    def calculate_outputs(self, inputs):

        network_outputs = []
        

        for i in range(len(self.structure) - 1):

            if (i == len(self.structure) - 2):

                is_final = True
                
            else:
                is_final = False
            
            outputs = self.calculate_outputs_single_layer(self.weights[i], self.biases[i], inputs, is_final)
            network_outputs.append(outputs)

            inputs = outputs
        
        outputs = inputs
        self.outputs = network_outputs
        return outputs

    def bp_pre_cost(self, final_outputs, desired_outputs):

        dcost_doutputs = []

        for output, desired_output in zip(final_outputs, desired_outputs) :
            dcost_doutput = 2*(output - desired_output)
            dcost_doutputs.append(dcost_doutput)

        return dcost_doutputs

    def bp_weights(self, input_layer, weights, dcost_doutputs):
        dcost_dweights = []

        for row, dcost_doutput in zip(weights, dcost_doutputs):
            
            dcost_drowweights = []

            for inputt in input_layer:
                dcost_dweight = dcost_doutput * inputt          # calculates it 
                dcost_drowweights.append(dcost_dweight)
            
            dcost_dweights.append(dcost_drowweights)

        return dcost_dweights
    
    
    def bp_inputs(self, input_layer, weights, dcost_doutputs):
    
        dcost_dinputs = []

        for i, inputt in enumerate(input_layer):

            dcost_dinput = 0

            for j, output in enumerate(dcost_doutputs):
                dcost_dinput += output * weights[j][i]
            
            dcost_dinputs.append(dcost_dinput)
        
        return dcost_dinputs
    

    def bp_RELU(self, pre_relu_nodes, post_relu_gradients):
    
        pre_relu_gradients = []

        for node, post_relu_gradient in zip(pre_relu_nodes, post_relu_gradients):
            if (node > 0):
                pre_relu_gradient = post_relu_gradient
            else:
                pre_relu_gradient = 0
            
            pre_relu_gradients.append(pre_relu_gradient)
        
        return pre_relu_gradients

    def bp_create_gradients(self, desired_outputs):

        all_dcost_dweights = []
        all_dcost_dbiases = []

        dcost_doutputs = self.bp_pre_cost(self.outputs[-1], desired_outputs)

        all_dcost_dbiases.append(dcost_doutputs) # add the last bias gradient

        for i in range(len(self.structure) - 2):
 
            dcost_dweights = self.bp_weights(self.outputs[-i - 2], self.weights[-i - 1], dcost_doutputs)

            all_dcost_dweights.append(dcost_dweights)   # add the weight gradients

            dcost_dinputs = self.bp_inputs(self.outputs[-i - 2], self.weights[-i - 1], dcost_doutputs)
            dcost_dinputs_pre_relu = self.bp_RELU(self.outputs[-i - 2], dcost_dinputs)

            all_dcost_dbiases.append(dcost_dinputs_pre_relu)   # add the bias gradients

            dcost_doutputs = dcost_dinputs_pre_relu

        
        dcost_dweights = self.bp_weights(self.outputs[0], self.weights[0], dcost_doutputs)
      
        all_dcost_dweights.append(dcost_dweights) # add the final weight gradient
           
        all_dcost_dweights = all_dcost_dweights[::-1]   #reverse weights list

        all_dcost_dbiases = all_dcost_dbiases[::-1]     # reverse bias list exept for the last element
        

        return (all_dcost_dweights, all_dcost_dbiases)

    def update_weights(self, big_weights_gradient):

        new_weights = []
        
        for old_weight_block, weight_gradient in zip(self.weights, big_weights_gradient):    #updates weights

            new_weight_block = []

            for i in range(len(old_weight_block)):

                new_weight_row = []

                for j in range(len(old_weight_block[i])):

                    new_weight = old_weight_block[i][j] - self.learning_rate * weight_gradient[i][j] #calculates new weight
                    new_weight_row.append(new_weight)

                new_weight_block.append(new_weight_row)
            
            new_weights.append(new_weight_block)
        
        self.weights = new_weights
        

    def update_biases(self, big_bias_gradient):
        
        # print("biases")
        # print(self.biases)

        # print("bias gradient")
        # print(big_bias_gradient)

        new_biases = []

        for old_biases_block, bias_gradient in zip(self.biases, big_bias_gradient):   #updates biases

            new_biases_block = []

            for i in range(len(old_biases_block)):
                new_bias = old_biases_block[i] - self.learning_rate * bias_gradient[i]   # calculates new bias
                new_biases_block.append(new_bias)
            
            new_biases.append(new_biases_block)

        self.biases = new_biases
    

    def train(self, num_epochs, data_file):

        with open(data_file, "r") as file:
            data = json.load(file)

        inputs = data["inputs"]
        target_outputs = data["target_outputs"]


        loss_values = []

        for epoch in range(num_epochs):
            
            total_loss = 0 

            print(" ")
            print("EPOCH #:", epoch + 1)
            print(" ")

            for inputt, target_output in zip(inputs, target_outputs):

                final_outputs = self.calculate_outputs(inputt)

                print("input:", inputt)
                print("output:", final_outputs)
                print("target_output:", target_output)
                print("")

                error = self.calculate_error(final_outputs, target_output)

                total_loss += error

                weights_gradient, biases_gradient = self.bp_create_gradients(target_output)
                
                self.update_weights(weights_gradient)
                self.update_biases(biases_gradient)
            
            average_loss = total_loss / len(inputs)
            print("LOSS:", average_loss)

            loss_values.append(average_loss)

        return loss_values
            

            
    


Network_Test = Network([1,10,10,1], 0.5, 0.0001)     # layer structure, weight varience, learning rate


loss_values = Network_Test.train(10000, "data.json")

test_output = Network_Test.calculate_outputs([6])
print("test output:", test_output)

plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


