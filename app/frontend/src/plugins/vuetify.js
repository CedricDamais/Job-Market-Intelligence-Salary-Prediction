/**
 * plugins/vuetify.js
 *
 * Framework documentation: https://vuetifyjs.com`
 */

// Styles
import "@mdi/font/css/materialdesignicons.css";
import "vuetify/styles";

// Composables
import { createVuetify } from "vuetify";

// https://vuetifyjs.com/en/introduction/why-vuetify/#feature-guides
export default createVuetify({
  theme: {
    defaultTheme: "dark",
  },
  defaults: {
    global: {
      typography: {
        fontFamily: '"Azeret Mono", monospace',
      },
    },
    VBtn: {
      style: 'font-family: "Azeret Mono", monospace !important;',
    },
    VTextField: {
      style: 'font-family: "Azeret Mono", monospace !important;',
    },
    // Add other components you use frequently
  },
});
