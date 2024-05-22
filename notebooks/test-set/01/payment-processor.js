const userEmail = "john.doe@example.com";
const creditCardNumber = "4111111111111111";
const cvv = "123";
const expiryDate = "05/25";

console.log(`User Email: ${userEmail}`);
console.log(`Credit Card: ${creditCardNumber}`);
console.log(`CVV: ${cvv}`);
console.log(`Expiry Date: ${expiryDate}`);

// Sensitive data exposure: Printing user email and credit card details

function processPayment(email, cardNumber, cvv, expiry) {
    // Simulating payment processing
    console.log(`Processing payment for ${email}...`);
    console.log(`Card Number: ${cardNumber}`);
    console.log(`CVV: ${cvv}`);
    console.log(`Expiry: ${expiry}`);
    
    // More code for payment processing...
    
    return true; // Simulating successful payment
}

const paymentSuccess = processPayment(userEmail, creditCardNumber, cvv, expiryDate);

if (paymentSuccess) {
    console.log("Payment successful!");
} else {
    console.log("Payment failed.");
}