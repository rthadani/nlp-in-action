(ns nlp-in-action.utils
  (:import (opennlp.tools.ngram NGramGenerator)
           (opennlp.tools.stemmer PorterStemmer)
           (opennlp.)))

(defn ngram-generator 
  [tokenizer sentence n & {:keys [seperator] :or {seperator " "}}]
  (let [strings (java.util.ArrayList. (tokenizer sentence))]
    (NGramGenerator/generate strings n seperator)))

(def stop-words
  (apply sorted-set ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]))

(defn remove-stop-words
  [words]
  (filter #(not (contains? stop-words (.toLowerCase %))) words))

(defn stem-words
  [words]
  (let [stemmer (PorterStemmer.)]
    (map #(.stem stemmer %) words)))


(defn lemmatizer
  [sentence]
  )

#_(remove-stop-words ["i" "I" "go" "to" "school"])

#_(stem-words  ["going" , "gone", "working", "development"])
