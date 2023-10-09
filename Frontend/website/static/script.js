function app() {
  return {
    step: 1,
    passwordStrengthText: "",
    togglePassword: false,

    password: "",
    gender: "Female",

    checkPasswordStrength() {
      var strongRegex = new RegExp(
        "^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*])(?=.{8,})"
      );
      var mediumRegex = new RegExp(
        "^(((?=.*[a-z])(?=.*[A-Z]))|((?=.*[a-z])(?=.*[0-9]))|((?=.*[A-Z])(?=.*[0-9])))(?=.{6,})"
      );

      let value = this.password;

      if (strongRegex.test(value)) {
        this.passwordStrengthText = "Strong password";
      } else if (mediumRegex.test(value)) {
        this.passwordStrengthText = "Could be stronger";
      } else {
        this.passwordStrengthText = "Too weak";
      }
    },
  };
}
