require 'porter2stemmer'
require 'tokenizer'
require 'numo/narray'
require "croupier"
require 'rambling-trie'

require "smalltext/version"

# probability threshold
ERROR_THRESHOLD = 0.2

module Smalltext
  class Error < StandardError; end
  # Your code goes here...

  class Classifier

  	def initialize
  		@training_data = []

  		#organizing our data structures for documents , @categories, words
  		@ignore_words = ['?']
  		@words=[]
		@categories=[]
		@documents=[]
		@tokenizer = Tokenizer::Tokenizer.new(:en)

		#create our bow training data
		@training=[]
		@output=[]
		@synapse = {}
  	end

  	def add_item(category, sentence)
  		@training_data.push({"category":category, "sentence":sentence})
  	end

  	def train(hidden_neurons=20, alpha=0.1, epochs=1000, dropout=false, dropout_percent=0.2)
  		preprocess
  		x_inp = Numo::NArray[training][0,true,true]
		y = Numo::NArray[output][0,true,true]

		start_time = Time.now

		neural_network(x_inp, y, hidden_neurons=hidden_neurons, alpha=alpha, epochs=epochs, dropout=dropout, dropout_percent=dropout_percent)

		elapsed_time = Time.now - start_time
		puts
		puts
		puts "Model training complete."
		puts "Processing time: #{elapsed_time} seconds"

  	end

	def classify(sentence, show_details=false)
	    results = think(sentence, show_details)
	    # puts "results is #{results.inspect}"

	    # results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
	    results = results.to_a.map.each_with_index {|r,i| [i, r] if r > ERROR_THRESHOLD }.compact 
	    # # results.sort(key=lambda x: x[1], reverse=True) 
	    results.sort! {|a,b| b[1] <=> a[1] }
	    # return_results =[[classes[r[0]],r[1]] for r in results]
	    return_results = results.map {|r| [klasses[r[0]], r[1]] }
	    puts "sentence: #{sentence}\nclassification: #{return_results}"
	    puts 
	    return return_results
	end

  	def save_model(synapse_file)
  		synapse_file = synapse_file

	    unless @synapse.empty?
		    File.open(synapse_file, 'wb') do |file|
		    	file.write(Marshal.dump(@synapse))
		    end
		    puts "saved synapses to: #{synapse_file}"
		else
			puts "Model not trained. Use the 'Classifier#train' method to build a model."
	    end	    
  	end

  	def load_model(synapse_file)
  		@synapse = Marshal.load(File.binread(synapse_file))
		@synapse[:synapse0] = Numo::NArray.cast(@synapse[:synapse0])
		@synapse[:synapse1] = Numo::NArray.cast(@synapse[:synapse1])

		@words = @synapse[:words]
		@categories = @synapse[:klasses]

		puts "Model #{synapse_file} loaded. Model was created on #{@synapse[:datetime]}"
  	end


  	private

  	def preprocess
		#loop through each sentence in our training data
		@training_data.each do |pattern|
		    #tokenize in each word in the sentence
		    w = @tokenizer.tokenize(pattern[:sentence])

		    #add to our words list
		    @words += w

		    #add to documents in our corpus
		    @documents.push([w,pattern[:category]])
		    
		    #add to our @categories list
		    if !@categories.include?(pattern[:category])
		        @categories.push(pattern[:category])
		    end
		end

		@ignore_words.each {|ign| @words.delete(ign) }
		@words.map! {|word| word.stem }
		@words.uniq!
		@categories.uniq!

		prepare_bow
  	end

  	def prepare_bow
  		#create an empty array for our output
		output_empty = Array.new(@categories.size) { 0 }

		#training set, bag of words for each sentence
		@documents.each do |doc|
		    #initialize our bag of words
		    bag=[]
		    #list of tokenized words for the pattern
		    pattern_words=doc[0]
		    #stem each word
		    pattern_words.map! {|word| word.stem }
		    #create our bag of words array
		    @words.each { |w| if pattern_words.include?(w) then bag << 1 else bag << 0 end }
		    @training.push(bag)
		    #output is a 0 for each tag and 1 for current tag
		    # output_row = Array.new(output_empty)
		    output_row = output_empty.dup
		    output_row[@categories.index(doc[1])] = 1
		    @output << output_row
		end
	end

	def training
	    return @training
	end

	def output
	    return @output
	end

	def klasses
	    return @categories
	end

	def words
	    return @words
	end

	def clean_up_sentence(sentence)
	    #tokenize the pattern
	    sentence_words = @tokenizer.tokenize(sentence)
	    #stem each word
	    # sentence_words=[stemmer.stem(word.lower()) for word in sentence_words]
	    sentence_words.map! {|word| word.stem }
	end

	#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
	def bow(sentence, words, show_details=false)
	    #tokenize the pattern
	    sentence_words=clean_up_sentence(sentence)
	    #bag of words
	    bag=[0] * words.size
	    # for s in sentence_words:
	    sentence_words.each do |s|        
	        words.each_with_index do |w,i|
	            if w == s
	                bag[i] = 1
	                if show_details
	                    puts "found in bag: #{w}"
	                end
	            end
	        end
	    end
	    # return Numo::Narray.new(bag)
	    return Numo::DFloat[bag].flatten
	end

	def think(sentence, show_details=false)
	    x= bow(sentence.downcase, words,show_details)
	    if show_details
	        puts "sentence: #{sentence},\nbow: #{x}"
	    end
	    #input layer is our bag of words
	    l0=x
	    # matrix multiplication of input and hidden layer
	    l1 = sigmoid(l0.dot @synapse[:synapse0])
	    # l1 = softmax(l0.dot @synapse_0)
	    # output layer
	    # l2 = sigmoid(l1.dot @synapse_1)
	    l2 = softmax(l1.dot @synapse[:synapse1])

	    return l2
	end

	def neural_network(x_inp, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=false, dropout_percent=0.5)

	    puts "Training with #{hidden_neurons} neurons, alpha:#{alpha}, dropout:#{dropout} #{dropout_percent if dropout}"
	    # puts x_inp.inspect
	    # puts "Input matrix: #{x_inp.size}x#{x_inp[0].size}    Output matrix: #{1}x#{@categories.size}"
	    puts "Input matrix: #{x_inp.shape}    Output matrix: #{1}x#{@categories.size}"
	    puts "Epochs set to #{epochs}. Every 100th iteration will be printed."
	    puts
	    
		last_mean_error = 1
	    # randomly initialize our weights with mean 0
	    # synapse_0 = 2*np.random.random((len(x_inp[0]), hidden_neurons)) - 1
	    synapse_0 = 2*Numo::DFloat.new(x_inp[0,true].size, hidden_neurons).rand - 1
	    # puts "synapse_0 is #{synapse_0.inspect}"
	    # synapse_1 = 2*np.random.random((hidden_neurons, len(@categories))) - 1
	    synapse_1 = 2*Numo::DFloat.new(hidden_neurons, @categories.size).rand - 1


	    prev_synapse_0_weight_update = synapse_0.new_zeros
	    prev_synapse_1_weight_update = synapse_1.new_zeros

	    synapse_0_direction_count = synapse_0.new_zeros
	    synapse_1_direction_count = synapse_1.new_zeros

	    (epochs + 1).times do |j|
	    	# Feed forward through layers 0, 1, and 2
	        layer_0 = x_inp
	        # puts "synapse_0 in block is #{synapse_0.inspect}"
	        # puts "layer_0 is #{layer_0.inspect}"
	        layer_1 = sigmoid(layer_0.dot synapse_0)
	        # layer_1 = tanh(layer_0.dot synapse_0)
	                
	        if dropout
	            # layer_1 *= np.random.binomial([np.ones((len(x_inp),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
	            b = Croupier::Distributions.binomial size: 1, success: (1-dropout_percent)
	            arr = Array.new(x_inp.size) { Array.new(hidden_neurons) {b.generate_number} }
	            layer_1 = Numo::DFloat[arr].reshape(x_inp.size,hidden_neurons) * (1.0/(1-dropout_percent))            
	        end

	        layer_2 = sigmoid((layer_1.dot synapse_1))
	        # layer_2 = tanh((layer_1.dot synapse_1))

	        # how much did we miss the target value?
	        layer_2_error = y - layer_2


	        if (j% 10000) == 0 and j > 5000
	            # if this 10k iteration's error is greater than the last iteration, break out
	            if (layer_2_error.abs).mean < last_mean_error
	                puts "delta after #{j} iterations: #{(layer_2_error.abs).mean} )"
	                last_mean_error = (layer_2_error.abs).mean
	            else
	                puts "break: #{(layer_2_error.abs).mean} > #{last_mean_error}"
	                break
	            end
	        end

	        # in what direction is the target value?
	        # were we really sure? if so, don't change too much.
	        # layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
	        layer_2_delta = layer_2_error * dtanh(layer_2)

	        # how much did each l1 value contribute to the l2 error (according to the weights)?
	        layer_1_error = layer_2_delta.dot(synapse_1.transpose)

	        # in what direction is the target l1?
	        # were we really sure? if so, don't change too much.
	        # layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
	        layer_1_delta = layer_1_error * dtanh(layer_1)
	        
	        synapse_1_weight_update = (layer_1.transpose).dot(layer_2_delta)
	        synapse_0_weight_update = (layer_0.transpose).dot(layer_1_delta)


	        if(j > 0)
	            # Bit array does not support arithmetic operation. Cast to Numo::Int32.cast, see https://github.com/ruby-numo/numo-narray/issues/65#issuecomment-323665534 
	            # puts "synapse_0_direction_count",synapse_0_direction_count.inspect
	            # puts "synapse_0_weight_update", synapse_0_weight_update.inspect
	            # puts "prev_synapse_0_weight_update", prev_synapse_0_weight_update.inspect
	            synapse_0_direction_count += ( Numo::Int32.cast((synapse_0_weight_update > 0)) - Numo::Int32.cast((prev_synapse_0_weight_update > 0)) ).abs
	            synapse_1_direction_count += ( Numo::Int32.cast((synapse_1_weight_update > 0)) - Numo::Int32.cast((prev_synapse_1_weight_update > 0))).abs
	        end
	        
	        synapse_1 += alpha * synapse_1_weight_update
	        synapse_0 += alpha * synapse_0_weight_update
	        
	        prev_synapse_0_weight_update = synapse_0_weight_update
	        prev_synapse_1_weight_update = synapse_1_weight_update
	        print "."
	        if (j%100 == 0)
	        	print j
	        end
	    end

	    now = Time.now
	    # puts "BEFORE DUMPING #{synapse_0.inspect}"
	    # persist synapses
	    @synapse = {'synapse0': synapse_0.to_a, 'synapse1': synapse_1.to_a,
	               'datetime': now.strftime("%Y-%m-%d %H:%M"),
	               'words': @words,
	               'klasses': @categories
	              }

	    # synapse_file = "intent_class.nn"

	    # File.open(synapse_file, 'wb') do |file|
	    # 	file.write(Marshal.dump(@synapse))
	    # end
	    # puts "saved synapses to: #{synapse_file}"
	end

	#compute sigmoid nonlinearity
	def sigmoid(x)
	    output=1/(1+Numo::NMath.exp(-x))    
	end
	#convert output of sigmoid function to its derivative
	def sigmoid_output_to_derivative(output)
	    output*(1-output)
	end

	# using softmax as output layer is recommended for classification where outputs are mutually exclusive
	def softmax(w)
	    e = Numo::NMath.exp(w - (w.max))
	    dist = e / (e.sum)
	    return dist
	end

	# using tanh over logistic sigmoid for the hidden layer is recommended   
	def tanh(x)
	     Numo::NMath.tanh(x)
	end

	# derivative for tanh sigmoid
	def dtanh(y)
	    # 1 - y*y
	    return 1.0 - Numo::NMath.tanh(y)**2
	end

  end # END class

end
