
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "smalltext/version"

Gem::Specification.new do |spec|
  spec.name          = "smalltext"
  spec.version       = Smalltext::VERSION
  spec.authors       = ["arjun"]
  spec.email         = ["arjunmenon009@gmail.com"]

  spec.summary       = %q{Classify short texts with neural network}
  spec.description   = %q{Classify short texts with neural network}
  spec.homepage      = "https://www.github.com/arjunmenon/smalltext"
  spec.license       = "MIT"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.
  if spec.respond_to?(:metadata)
    spec.metadata["allowed_push_host"] = "TODO: Set to 'http://mygemserver.com'"

    spec.metadata["homepage_uri"] = spec.homepage
    spec.metadata["source_code_uri"] = "https://www.github.com/arjunmenon/smalltext"
    spec.metadata["changelog_uri"] = "https://www.github.com/arjunmenon/smalltext/CHANGELOG.md"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files         = Dir.chdir(File.expand_path('..', __FILE__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_runtime_dependency 'rambling-trie'
  spec.add_runtime_dependency 'croupier'
  spec.add_runtime_dependency 'numo-narray', '~> 0.9.1.3'
  spec.add_runtime_dependency 'tokenizer'
  spec.add_runtime_dependency 'porter2stemmer'

  spec.add_development_dependency "bundler", "~> 1.17"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rspec", "~> 3.0"
end
