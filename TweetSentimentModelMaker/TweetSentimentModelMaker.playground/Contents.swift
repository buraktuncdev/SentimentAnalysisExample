import Cocoa
import CreateML


let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/buraktunc/Desktop/buraktuncdev/twitter-sanders-apple3.csv"))

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evaluationMetrics = sentimentClassifier.evaluation(on: trainingData, textColumn: "text", labelColumn: "class")

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "Burak Tunç", shortDescription: "Model trained for sentiment Tweets", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/buraktunc/Desktop/buraktuncdev/twitter-sanders-apple3.csv"))

try sentimentClassifier.prediction(from: "@Twitter is a terrible company!")

try sentimentClassifier.prediction(from: "I just found the best restaurant ever, and it's @McDonalds")

try sentimentClassifier.prediction(from: "I think @CocaCola ads are jusk ok.")
