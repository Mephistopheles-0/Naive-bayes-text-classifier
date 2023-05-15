using Statistics

# Define a function to train the Bernoulli Naive Bayes classifier
function train_bernoulli_naive_bayes(documents, labels)
    # Get the unique labels and calculate the prior probabilities
    unique_labels = unique(labels)
    prior_probs = [sum(labels .== label) / length(labels) for label in unique_labels]

    # Split the documents into separate words and count the occurrences of each word for each label
    word_counts = Dict{String, Vector{Int}}()
    for i in 1:length(documents)
        words = split(documents[i], " ")
        label = labels[i]
        if !haskey(word_counts, label)
            word_counts[label] = zeros(Int, length(unique(words)))
        end
        for word in words
            if !(word in keys(word_counts[label]))
                word_counts[label][word] = 0
            end
            word_counts[label][word] += 1
        end
    end

    # Calculate the conditional probabilities for each word and label combination
    word_probs = Dict{String, Vector{Float64}}()
    for (label, counts) in word_counts
        total_count = sum(counts)
        word_probs[label] = (counts .+ 1) ./ (total_count + length(counts))
    end

    # Return the trained model
    return (unique_labels, prior_probs, word_probs)
end

# Define a function to classify new documents using the trained Bernoulli Naive Bayes model
function classify_bernoulli_naive_bayes(document, unique_labels, prior_probs, word_probs)
    # Split the document into separate words and calculate the conditional probabilities for each label
    words = split(document, " ")
    label_probs = zeros(length(unique_labels))
    for i in 1:length(unique_labels)
        label = unique_labels[i]
        label_probs[i] = log(prior_probs[i])
        for word in words
            if haskey(word_probs[label], word)
                label_probs[i] += log(word_probs[label][word])
            else
                label_probs[i] += log(1 / (sum(word_probs[label]) + length(word_probs[label])))
            end
        end
    end

    # Determine the label with the highest conditional probability
    max_label_index = argmax(label_probs)
    return unique_labels[max_label_index]
end