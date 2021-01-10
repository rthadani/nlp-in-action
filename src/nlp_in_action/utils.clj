(ns nlp-in-action.utils
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random])
  (:import (opennlp.tools.ngram NGramGenerator)
           (opennlp.tools.stemmer PorterStemmer)
           (java.io DataInputStream)
           (java.nio ByteBuffer ByteOrder)))

(defn ngram-generator 
  [tokenizer sentence n & {:keys [seperator] :or {seperator " "}}]
  (let [strings (java.util.ArrayList. (tokenizer sentence))]
    (NGramGenerator/generate strings n seperator)))

(def stop-words
  (apply sorted-set ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
                     "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                     "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", 
                     "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                     "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
                     "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", 
                     "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
                     "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]))

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


(def w2v-file-path "../../data/GoogleNews-vectors-negative300.bin") ;; the word2vec file path
(def EOS "</s>")  ;; end of sentence word

(defn r-string
  "Reads a string from the given DataInputStream `dis` until a space or newline is reached."
  [dis]
  (loop [b (.readByte dis)
         bs []]
    (if (and (not= 32 b) (not= 10 b))
      (recur (.readByte dis) (conj bs b))
      (new String (byte-array bs)))))

(defn get-float [bs]
  (-> (ByteBuffer/wrap bs)
      (.order ByteOrder/LITTLE_ENDIAN)
      (.getFloat)))

(defn read-float [is]
  (let [bs (byte-array 4)]
    (do (.read is bs)
        (get-float bs))))

(defn- load-w2v-vectors
  "Lazily loads the word2vec vectors given a data input stream `dis`,
  number of words `nwords` and dimensionality `embedding-size`."
  [dis embedding-size num-vectors]
  (if (= 0 num-vectors)
    (list)
    (let [word (r-string dis)
          vect (mapv (fn [_] (read-float dis)) (range embedding-size))]
      (cons [word vect] (lazy-seq (load-w2v-vectors dis embedding-size (dec num-vectors)))))))

(defn load-word2vec-model!
  "Loads the word2vec model stored in a binary format from the given `path`.
  By default only the first 100 embeddings are loaded."
  ([path embedding-size opts]
   (println "Loading the word2vec model from binary ...")
   (with-open [bis (io/input-stream path)
               dis (new DataInputStream bis)]
     (let [word-size (Integer/parseInt (r-string dis))
           dim  (Integer/parseInt (r-string dis))
           {:keys [max-vectors vocab] :or {max-vectors word-size}} opts
           _  (println "Processing with " {:dim dim :word-size word-size} " loading max vectors " max-vectors)
           _ (if (not= embedding-size dim)
               (throw (ex-info "Mismatch in embedding size"
                               {:input-embedding-size embedding-size
                                :word2vec-embedding-size dim})))
           vectors (load-w2v-vectors dis dim max-vectors)
           word2vec (if vocab
                      (->> vectors
                           (filter (fn [[w _]] (contains? vocab w)))
                           (into {}))
                      (->> vectors
                           (take max-vectors)
                           (into {})))]
       (println "Finished")
       {:num-embed dim :word2vec word2vec})))
  ([path embedding-size]
   (load-word2vec-model! path embedding-size {:max-vectors 100})))

(defn read-text-embedding-pairs [pairs]
  (for [^String line pairs
        :let [fields (.split line " ")]]
    [(aget fields 0)
     (mapv #(Float/parseFloat ^String %) (rest fields))]))

(defn clean-str [s]
  (-> s
      (string/replace #"^A-Za-z0-9(),!?'`]" " ")
      (string/replace #"'s" " 's")
      (string/replace #"'ve" " 've")
      (string/replace #"n't" " n't")
      (string/replace #"'re" " 're")
      (string/replace #"'d" " 'd")
      (string/replace #"'ll" " 'll")
      (string/replace #"," " , ")
      (string/replace #"!" " ! ")
      (string/replace #"\(" " ( ")
      (string/replace #"\)" " ) ")
      (string/replace #"\?" " ? ")
      (string/replace #" {2,}" " ")
      (string/trim)))

(defn load-mr-data-and-labels
  "Loads MR polarity data from files, splits the data into words and generates labels. 
  Returns split sentences and labels."
  [path max-examples]
  (println "Loading all the movie reviews from " path)
  (let [positive-examples (mapv #(string/trim %) (-> (file-seq  (str path "/pos"))
                                                   (map slurp)))
        negative-examples (mapv #(string/trim %) (-> (file-seq  (str path "/neg"))
                                                   (map slurp)))
        positive-examples (into [] (if max-examples (take max-examples positive-examples) positive-examples))
        negative-examples (into [] (if max-examples (take max-examples negative-examples) negative-examples))
        ;; split by words
        x-text (->> (into positive-examples negative-examples)
                    (mapv clean-str)
                    (mapv #(string/split % #" ")))

        ;; generate labels
        positive-labels (mapv (constantly 1) positive-examples)
        negative-labels (mapv (constantly 0) negative-examples)]
    {:sentences x-text :labels (into positive-labels negative-labels)}))

(defn pad-sentences
  "Pads all sentences to the same length where the length is defined by the longest sentence. Returns padded sentences."
  [sentences]
  (let [padding-word EOS
        sequence-len (apply max (mapv count sentences))]
    (mapv (fn [s] (let [diff (- sequence-len (count s))]
                    (if (pos? diff)
                      (into s (repeat diff padding-word))
                      s)))
          sentences)))

(defn build-vocab-embeddings
  "Returns the subset of `embeddings` for words from the `vocab`.
  Embeddings for words not in the vocabulary are initialized randomly
  from a uniform distribution."
  [vocab embedding-size embeddings]
  (into {}
        (mapv (fn [[word _]]
                [word (or (get embeddings word)
                          (ndarray/->vec (random/uniform -0.25 0.25 [embedding-size])))])
              vocab)))

(defn build-input-data-with-embeddings
  "Map sentences and labels to vectors based on a pretrained embeddings."
  [sentences embeddings]
  (mapv (fn [sent] (mapv #(embeddings %) sent)) sentences))

(defn build-vocab
  "Creates a vocabulary for the data set based on frequency of words.
  Returns a map from words to unique indices."
  [sentences]
  (let [words (flatten sentences)
        wc (reduce
            (fn [m w] (update-in m [w] (fnil inc 0)))
            {}
            words)
        sorted-wc (sort-by second > wc)
        sorted-w (map first sorted-wc)]
    (into {} (map vector sorted-w (range (count sorted-w))))))

(defn load-ms-with-embeddings
  "Loads the movie review sentences data set for the given
  `:pretrained-embedding` (e.g. `nil`, `:glove` or `:word2vec`)"
  [path max-examples embedding-size ]
  (let [{:keys [sentences labels]} (load-mr-data-and-labels path max-examples)
        sentences-padded  (pad-sentences sentences)
        vocab (build-vocab sentences-padded)
        vocab-embeddings (->> (load-word2vec-model! w2v-file-path embedding-size {:vocab vocab})
                       (:word2vec)
                       (build-vocab-embeddings vocab embedding-size))
        data (build-input-data-with-embeddings sentences-padded vocab-embeddings)]
    {:data data
     :label labels
     :sentence-count (count data)
     :sentence-size (count (first data))
     :embedding-size embedding-size
     :vocab-size (count vocab) }))
