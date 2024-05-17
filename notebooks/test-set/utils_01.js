// Helper function to generate a random string
function generateRandomString(length) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    for (let i = 0; i < length; i++) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
    return result;
  }
  
  // Helper function to generate a random number within a range
  function generateRandomNumber(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }
  
  // Helper function to shuffle an array
  function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
  
  // Helper function to calculate the sum of elements in an array
  function calculateArraySum(array) {
    return array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
  }
  
  // Helper function to calculate the average of elements in an array
  function calculateArrayAverage(array) {
    const sum = calculateArraySum(array);
    return sum / array.length;
  }
  
  // Helper function to find the maximum value in an array
  function findMaxValue(array) {
    return Math.max(...array);
  }
  
  // Helper function to find the minimum value in an array
  function findMinValue(array) {
    return Math.min(...array);
  }
  
  // Helper function to check if a value exists in an array
  function valueExists(array, value) {
    return array.includes(value);
  }
  
  // Helper function to remove duplicate values from an array
  function removeDuplicates(array) {
    return [...new Set(array)];
  }
  
  // Helper function to count the occurrences of a value in an array
  function countOccurrences(array, value) {
    return array.filter((v) => v === value).length;
  }
  
  // Helper function to merge two arrays and remove duplicates
  function mergeArrays(array1, array2) {
    const mergedArray = [...array1, ...array2];
    return removeDuplicates(mergedArray);
  }
  
  // Helper function to flatten a nested array
  function flattenArray(array) {
    return array.reduce((acc, val) => acc.concat(Array.isArray(val) ? flattenArray(val) : val), []);
  }
  
  // Helper function to convert a string to camel case
  function toCamelCase(str) {
    return str.replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase();
    }).replace(/\s+/g, '');
  }
  
  // Helper function to convert a string to kebab case
  function toKebabCase(str) {
    return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
  }
  
  // Helper function to truncate a string and add an ellipsis
  function truncateString(str, maxLength) {
    if (str.length <= maxLength) {
      return str;
    }
    return `${str.slice(0, maxLength)}...`;
  }
  
  // Helper function to capitalize the first letter of a string
  function capitalizeFirstLetter(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
  
  // Helper function to delay execution of a function
  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
  
  // Helper function to debounce a function
  function debounce(fn, delay) {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        fn.apply(null, args);
      }, delay);
    };
  }
  
  // Helper function to throttle a function
  function throttle(fn, delay) {
    let lastCall = 0;
    return (...args) => {
      const now = new Date().getTime();
      if (now - lastCall < delay) {
        return;
      }
      lastCall = now;
      return fn.apply(null, args);
    };
  }
  
  // Helper function to format a date
  function formatDate(date, format) {
    const map = {
      mm: date.getMonth() + 1,
      dd: date.getDate(),
      yy: date.getFullYear().toString().slice(-2),
      yyyy: date.getFullYear(),
    };
  
    return format.replace(/mm|dd|yy|yyyy/gi, (matched) => map[matched].toString().padStart(2, '0'));
  }
  
  // Helper function to get the current timestamp
  function getTimestamp() {
    return new Date().getTime();
  }
  
  // Helper function to generate a unique identifier
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }
  
  // Helper function to check if a value is a valid email address
  function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
  
  // Helper function to check if a value is a valid URL
  function isValidUrl(url) {
    const urlRegex = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/;
    return urlRegex.test(url);
  }
  
  // Helper function to check if a value is a valid phone number
  function isValidPhoneNumber(phoneNumber) {
    const phoneRegex = /^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$/im;
    return phoneRegex.test(phoneNumber);
  }
  
  // Helper function to convert a string to a slug
  function toSlug(str) {
    return str
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/(^-|-$)+/g, '')
      .slice(0, 200);
  }
  
  // Helper function to truncate a string at the first whitespace after a specified length
  function truncateAtWhitespace(str, length) {
    if (str.length <= length) {
      return str;
    }
    const truncated = str.slice(0, length);
    return `${truncated.trim()}...`;
  }
  
  // Helper function to format a number with commas as thousands separators
  function formatNumberWithCommas(number) {
    return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }
  
  // Helper function to check if a value is a number
  function isNumber(value) {
    return typeof value === 'number' && isFinite(value);
  }
  
  // Helper function to check if a value is an integer
  function isInteger(value) {
    return isNumber(value) && value % 1 === 0;
  }
  
  // Helper function to check if a value is a positive number
  function isPositiveNumber(value) {
    return isNumber(value) && value >= 0;
  }
  
  // Helper function to check if a value is a negative number
  function isNegativeNumber(value) {
    return isNumber(value) && value < 0;
  }
  
  // Helper function to check if a value is an even number
  function isEvenNumber(value) {
    return isInteger(value) && value % 2 === 0;
  }
  
  //