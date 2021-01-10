(ns nlp-in-action.chapter3)


(def build-one-row
  [document-tf columns]
  )

(defn build-doc-tf
  [document tokenizer]
  (let [frequency-word-builder (comp frequencies u/remove-stop-words tokenizer)]
    (->> (frequency-word-builder document)
        (map (fn [word frequency] [word (/ frequency (count document))]))
        (into {}))))


(defn build-word-vector
  [document-tfs]
  (let [columns (reduce (fn [s tf] (into s (keys tf))) document-tfs)]
    ))
