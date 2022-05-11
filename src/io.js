import * as tf from "@tensorflow/tfjs-node";
import jpeg from "jpeg-js";
import { dataset } from "./dataset.js";
import * as fs from "fs";

// Create dataset array
const datasetArray = await dataset.toArray();

// Create generator function with inputs of training and validation data
function* getInputs() {
  for (let i = 0; i < datasetArray.length; i++) {
    const imageLocation =
      "/Users/alessiotoniolo/Documents/BSA/BirdModel/assets/" +
      datasetArray[i].filepaths;

    const imageData = jpeg.decode(fs.readFileSync(imageLocation), {
      useTArray: true,
    });

    yield Array.from(imageData.data);

    //const imageData = tf.node.decodeJpeg(fs.readFileSync(imageLocation), 3);

    //yield imageData;
  }
}

// Create a dataset from the generator function
const inputsDataset = tf.data.generator(getInputs);

//inputsDataset.forEachAsync((e) => console.log(e));

// Create outputs array
let outputs = [];

// Add class indicies to outputs array which maps to the inputs
await dataset.forEachAsync((element) => {
  outputs.push(element.classindex);
});

const outputsTensor = tf.oneHot(tf.tensor1d(outputs, "int32"), 400);

const outputsArray = await outputsTensor.array();

/*
function* getOutputs() {
  for (let i = 0; i < datasetArray.length; i++) {
    const encodedOutputs = tf.tensor1d(outputsArray[i]);

    yield encodedOutputs;
  }
}

//const yDataset = tf.data.array(outputsArray);

const yDataset = tf.data.generator(getOutputs);
*/
//yDataset.forEachAsync((e) => console.log(e));

const yDataset = tf.data.array(outputsArray);

//outputs.map((val) => [val]);

//yDataset.forEachAsync((e) => console.log(e));

//const yDataset = tf.data.array(outputs);

const xyDataset = tf.data.zip({ xs: inputsDataset, ys: yDataset });

//await xyDataset.forEachAsync((e) => console.log(e));

export { xyDataset };
