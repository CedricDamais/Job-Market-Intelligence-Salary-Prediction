<template>
  <v-container class="w-100 h-100 bg-blue-grey-darken-1 pa-0 ma-0" fluid>
    <v-row>
      <v-col>
        <div class="pa-5">
        <span>
          <h1>NLP Project - LinkedIn Job Offers</h1>
        </span>
        <span>
          <h3 class="font-weight-light font-italic">Try out our different models here</h3>
        </span>
      </div>
        <v-row class="mt-10 ml-8">
          <v-col cols="4">
        <SidebarComponent @update-regression="handleRegressionUpdate" @update-generation="handleGenerationUpdate" class="w-100 h-100"/>
      </v-col>
        <v-col cols="7" class="ml-5 mr-8">
          <MainContentComponent :selectedRegression="selectedRegression" :selectedGeneration="selectedGeneration" class="w-100 h-100"/>
        </v-col>
      </v-row>
      </v-col>
    </v-row>
</v-container>
</template>

<script setup>
  import SidebarComponent from '@/components/SidebarComponent.vue'
  import MainContentComponent from '@/components/MainContentComponent.vue'
  import { ref, watch } from 'vue';

  const selectedRegression = ref("");
  const selectedGeneration = ref("");

  const handleRegressionUpdate = (model) => {
    selectedRegression.value = model;
    selectedGeneration.value = "";
  };

  const handleGenerationUpdate = (model) => {
      selectedGeneration.value = model;
      selectedRegression.value = "";
  }
</script>
