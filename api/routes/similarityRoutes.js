'use strict';
module.exports = function(app) {
  var similarity = require('../controllers/similarityController');

  // todoList Routes
  app.route('/similarity')
    // .get(todoList.list_all_tasks)
    .post(similarity.create_a_task);


  // app.route('/similarity/:taskId')
  //   .get(todoList.read_a_task)
  //   .put(todoList.update_a_task)
  //   .delete(todoList.delete_a_task);
};
