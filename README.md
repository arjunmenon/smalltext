# Smalltext

Classify short texts with neural network.

This gem is specifically created to classify small sentence/datasets using a supervised training algorithm. You can use this in place of Naive Bayes.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'smalltext'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install smalltext


## Dependencies

Gem depends on Numo/NArray, Porter2Stemmer, Tokenizer, Croupier

Gem dependencies should be automatically installed.

## Usage

Classification is easy to get started.

```ruby
require 'smalltext'

s = Smalltext::Classifier.new

# Add your sentence using the `add_item` method
# Classifier#add_item(category, sentence)

s.add_item("schedule_list", "What time is my next session?")
s.add_item("schedule_list", "When is my next session?")
s.add_item("schedule_list", "What time is my next meeting?")
s.add_item("schedule_list", "Can you please show me my schedule?")
s.add_item("schedule_list", "Show me my schedule.")

s.add_item("greetings", "Hi")
s.add_item("greetings", "How are you doing?")
s.add_item("greetings", "have a nice day")
s.add_item("greetings", "good morning.")
s.add_item("greetings", "Whats up")
s.add_item("greetings", "Yo")

s.add_item("where_is", "Where is narkel bagan")
s.add_item("where_is", "show me the way to sasta sundar")
s.add_item("where_is", "where is the staircase")
s.add_item("where_is", "give me the direction to the parking lot")

s.add_item("weather", "Whats the weather")
s.add_item("weather", "Weather in Noida")
s.add_item("weather", "Is it raining")
s.add_item("weather", "Will it be hot tomorrow in Mumbai")
s.add_item("weather", "What is the maximum temperature today")

s.add_item("finance", "How many dollars is 17 euros?")
s.add_item("finance", "How much is 100 ruppees in US dollars")
s.add_item("finance", "How much is Starbucks stock?")
s.add_item("finance", "Tell me bitcoin exchange rate")
s.add_item("finance", "What is the value of ruppee")
s.add_item("finance", "Share price of Microsoft")

# Train a model using the Classifier#train method

s.train

# Test your trained model using the Classifier#classify(sentence) method

s.classify("give me the direction to moon")

# sentence: give me the direction to moon
# classification: [["where_is", 1.0]]

# => [["where_is", 1.0]]
```

You can also save your model. Use the `Classifier#save_model(file_name)`

```ruby
s.save_model('intents.model')
```

To load a saved model use the `Classifier#load_model(file_name)`

```ruby
s.load_model('intents.model')

s.classify("when is the next meeting")
# sentence: when is the next meeting
# classification: [["schedule_list", 0.9999189960209529]]

# => [["schedule_list", 0.9999189960209529]]
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/arjunmenon/smalltext. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the Smalltext project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/[USERNAME]/smalltext/blob/master/CODE_OF_CONDUCT.md).

## Credits

This gem is a Ruby port to the [Medium article](https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6) describing text classification in Python.

## Roadmap

Goal of this gem is be an efficient tool for short texts classifications, where dataset is a constraint.

- Implement word vectors.
When dataset is sparse, we should leverage word vectors to map in categorizing unseen words.

- More algorithm options
Apart from neural networks, one can also switch and compare with different algorithms which mostly suits their needs.

## Todo

- Write Tests
- Add support for batch training
- Add SVD
- Add support for word vectors. Need to try this [method](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/).
- Add more classification algorithms.
- Add benchmarking
- Create model for known text datasets like Reuters, etc.
