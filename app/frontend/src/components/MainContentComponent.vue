<template>
  <v-card class="bg-blue-grey-darken-4">
        <v-card-title v-if="selectedModel == ''" class="font-italic font-weight-light">
          Please select a model
        </v-card-title>
        <v-card-title v-else>
          {{ selectedModel }}
          <v-card-text>
          <div>

            <h4>
            Input
          </h4>
            <v-textarea label="Input" class="mt-2 ml-n1" clearable v-model="jobTitle"></v-textarea>
            <v-btn
              class="mt-2"
              color="black"
              @click="generateDescription"
              :disabled="loading || !jobTitle"
            >
              Run Model
            </v-btn>
        </div>
        <div class="mt-4">
          <h4>
            Output
          </h4>
          <v-alert v-if="error" type="error" class="mt-2">{{ error }}</v-alert>
          <v-textarea
            v-if="description" 
            v-model="description"
            label="Generated Description"
            readonly
            class="mt-2"
            auto-grow
            rows="5"
          ></v-textarea>
        </div>
        </v-card-text>
        </v-card-title>
  </v-card>
</template>

<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  selectedRegression: {
    type: String,
    default: "",
  },
  selectedGeneration: {
    type: String,
    default: "",
  },
});

const selectedModel = computed(() => {
  if (props.selectedRegression != "") {
    return props.selectedRegression;
  } else if (props.selectedGeneration != "") {
    return props.selectedGeneration;
  }
  return "";
});

const jobTitle = ref("");
const description = ref("");
const loading = ref(false);
const error = ref("");

async function generateDescription() {
  console.log("generateDescription function called!");
  error.value = "";
  description.value = "";
  loading.value = true;
  try {
    const response = await fetch("http://127.0.0.1:8000/lstm_job_description", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: jobTitle.value }),
    });
    if (!response.ok) {
      console.error("Backend response error:", response.status, await response.text());
      throw new Error(`Backend error: ${response.statusText}`);
    } 
    const data = await response.json();
    description.value = data.description;
  } catch (e) {
    console.error("Error during fetch:", e);
    error.value = e.message || "Failed to fetch description.";
  } finally {
    loading.value = false;
  }
}

</script>
