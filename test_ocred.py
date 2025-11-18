"""Quick test to verify OcredFragmentModel and OcredFragmentCorpus work correctly."""

from ebl_ngrams import OcredFragmentModel, OcredFragmentCorpus, DEFAULT_N_VALUES

# Test that classes can be imported
print("✓ Successfully imported OcredFragmentModel and OcredFragmentCorpus")

# Test instantiation with sample data
test_data = [
    {"_id": "test.1", "ocredSigns": "A B C\nD E F"},
    {"_id": "test.2", "ocredSigns": "G H I\nJ K L"},
]

corpus = OcredFragmentCorpus(test_data, n_values=DEFAULT_N_VALUES)
print(f"✓ Created OcredFragmentCorpus with {len(corpus)} fragments")

# Test fragment property
print(f"✓ Corpus has {len(corpus.fragments)} fragments accessible via .fragments property")

# Test model instantiation
model = OcredFragmentModel("test.3", "M N O\nP Q R", n_values=DEFAULT_N_VALUES)
print(f"✓ Created OcredFragmentModel with {len(model)} ngrams")

# Test match functionality
matches = corpus.match(model)
print(f"✓ Match operation works, returned {len(matches)} results")

print("\n✅ All basic functionality tests passed!")
