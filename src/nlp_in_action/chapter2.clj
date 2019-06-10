(ns nlp-in-action.chapter2
  (:require [clojure.string :as str] 
            [org.apache.clojure-mxnet.ndarray :as nd]
            [opennlp.nlp :as n]
            [nlp-in-action.utils :as u]))


(defn build-single-row
  [word-map bag-of-word-positions]
  (let [row (transient (vec (repeat (count bag-of-word-positions) 0)))]
    (doseq [[k v] word-map]
      (assoc! row (get bag-of-word-positions k) v))
    (nd/array (persistent! row) [1 (count bag-of-word-positions)])))

(defn bag-of-words
  [sentences tokenizer]
  (println sentences)
  (let [all-words-map (map (comp frequencies u/remove-stop-words tokenizer) sentences)
        distinct-words (reduce (fn [s sentence-map] (into s (keys sentence-map))) 
                               (sorted-set) 
                               all-words-map)
        bag-of-word-positions (first (reduce 
                                      (fn [[r i] w] [(assoc r w i) (inc i)]) 
                                      [{} 0]
                                      distinct-words))]
    (->> all-words-map
        (map (fn [word-map] (build-single-row word-map bag-of-word-positions)))
        vec)))

(def sentences
  ["Thomas Jefferson began building Monticello at the age of 26."
    "Construction was done mostly by local masons and carpenters."
    "He moved into the South Pavilion in 1770."
    "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."])
#_(->> sentences
     (map (comp frequencies #(str/split % #"[-\s.,;!?]+")))
     (reduce (fn [s sentence-map] (into s (keys sentence-map))) (sorted-set)))

#_ (def bw (bag-of-words sentences  #(str/split % #"[-\s.,;!?]+")))
#_ (map (comp frequencies (n/make-tokenizer "models/en-token.bin")) sentences)
#_ (map (comp frequencies #(u/ngram-generator (n/make-tokenizer "models/en-token.bin") % 2)) sentences)
#_ (def bw1 (bag-of-words sentences (n/make-tokenizer "models/en-token.bin")) )

#_ (nd/->vec (nd/dot (bw 0) (nd/transpose (bw 3))))
#_ (nd/->vec (nd/dot (bw1 0) (nd/transpose (bw1 3))))
