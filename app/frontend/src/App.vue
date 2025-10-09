<template>
  <v-app>
    <v-main>
      <router-view />
      <!-- Move your form inside the template! -->
      <v-textarea
        v-model="jobTitle"
        label="Job Title"
        class="mt-2 ml-n1"
        clearable
      ></v-textarea>
      <v-btn
        class="mt-2"
        color="primary"
        @click="generateDescription"
        :disabled="loading || !jobTitle"
      >
        Generate Description
      </v-btn>
      <div class="mt-4">
        <h4>Output</h4>
        <v-alert v-if="error" type="error" class="mt-2">{{ error }}</v-alert>
        <v-textarea
          v-if="description"
          v-model="description"
          label="Generated Description"
          readonly
          class="mt-2"
        ></v-textarea>
      </div>
    </v-main>
  </v-app>
</template>

<script setup>
  import { ref, computed } from 'vue';

const props = defineProps({
  selectedRegression: { type: String, default: "" },
  selectedGeneration: { type: String, default: "" },
});

const selectedModel = computed(() => {
  if (props.selectedRegression != "") return props.selectedRegression;
  if (props.selectedGeneration != "") return props.selectedGeneration;
  return "";
});

const jobTitle = ref("");
const description = ref("");
const loading = ref(false);
const error = ref("");

async function generateDescription() {
  console.log("Generating description");
  error.value = "";
  description.value = "";
  loading.value = true;
  try {
    const response = await fetch("http://127.0.0.1:8000/lstm_job_description", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: jobTitle.value }),
    });
    if (!response.ok) throw new Error("Backend error");
    const data = await response.json();
    description.value = data.description;
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}
</script>