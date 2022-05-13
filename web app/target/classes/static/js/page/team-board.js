//var boardId =  getUrlParam("boardId");
var boardId =  $("#boardId").val();
//alert(boardId);
layui.use(['form','layuimini'], function () {
    var form = layui.form,
        layer = layui.layer,
        layuimini = layui.layuimini;
    form.render();
});
var AddList = function(){
	layer.open({
	    title:"Add another list",
	    btn: ['confirm', 'cancel'],
	    btnAlign: 'c',
	    offset: '80px',
	    area: ['400px', '200px'],
	    content: '<input type="text" name="name" id="name" required  lay-verify="required" placeholder="name" autocomplete="off" class="layui-input">'
    	,yes: function(index, layero){
    	    //按钮【按钮一】的回调
    	    var name = $("#name").val();
    	    if(name==null || name==""){
    	    	layer.msg("enter name");
    	    	return false;
    	    }
    	    $.post("/private/board/list/save",{boardId:boardId,name:name},function(data){
    	    	layer.msg(data.message,function(){
    	    	  if(data.success){
  					  layer.closeAll();
  					  //refreshTab();
  					window.location.reload(); 
  				  }
  				  layer.close(index);
    	    	});
    	    });
    	 }
	    ,cancel: function(){ 
	    }
	  });
}
var addCard = function(listId){
	$.post('/private/board/list/card/add', {listId:listId}, function(str){
	  layer.open({
	    type: 1,
	    title:"Add another card",
	    area: ['600px', '400px'],
	    content: str 
	  });
	});
}
var editCard = function(cardId){
	$.post('/private/board/list/card/edit', {cardId:cardId}, function(str){
	  layer.open({
	    type: 1,
	    title:"edit card",
	    area: ['800px', '800px'],
	    content: str 
	  });
	});
}