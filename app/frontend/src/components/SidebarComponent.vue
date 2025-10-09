<template>
  <v-card class="bg-blue-grey-darken-4">
    <v-card-title>Menu</v-card-title>
    <v-card-text>
      <span>
        <p>Choose your model to try it out</p>
      </span>
      <v-select v-model="selectedRegression" :items="regressionModels" class="mt-8 w-75" density="comfortable" label="Regression models">
      </v-select>
      
      <v-select v-model="selectedGeneration" :items="generationModels" class="mt-5 w-75" density="comfortable" label="Generation models">
      </v-select>
    </v-card-text>
  </v-card>
</template>

<script setup>
import { ref, watch } from 'vue';

const emit = defineEmits(['update-regression', 'update-generation']);
const regressionModels = [
  'Linear Regression',
  'Random Forest Regression',
  'Feed Forward Neural Network (FFNN)',
  'Recurrent Neural Network (RNN)',
  'Transformer',
];
const generationModels = [
  'N-Gram',
  'Feed Forward Neural Network (FFNN)',
  'Recurrent Neural Network (RNN)',
  'Transformer',
];
const selectedRegression = ref(null);
const selectedGeneration = ref(null);

// Event listeners
watch(selectedRegression, (newValue) => {
if (newValue) {
  selectedGeneration.value = null; // Reset generation model when a regression model is selected
  emit('update-generation', "");
  emit('update-regression', newValue);

}

});

watch(selectedGeneration, (newValue) => {
  if (newValue) {
    selectedRegression.value = null; // Reset generation model when a regression model is selected
    emit('update-regression', "");
    emit('update-generation', newValue);

  }
  // Emit the selected model to the parent component
});

</script>
