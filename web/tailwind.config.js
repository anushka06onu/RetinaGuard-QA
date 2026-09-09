/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          500: '#14b8a6',
          700: '#0f766e',
          900: '#134e4a',
          navy: '#0f172a',
          slate: '#1e293b'
        }
      },
      fontFamily: {
        sans: ['Inter', 'Source Sans 3', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
