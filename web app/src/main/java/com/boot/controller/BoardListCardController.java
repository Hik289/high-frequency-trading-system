package com.boot.controller;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.propertyeditors.CustomDateEditor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.WebDataBinder;
import org.springframework.web.bind.annotation.InitBinder;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.boot.entity.BoardList;
import com.boot.entity.BoardListCard;
import com.boot.entity.Files;
import com.boot.service.BoardListCardService;
import com.boot.service.BoardService;
import com.boot.service.FilesService;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
/**
 * board
 * card
 */
@Controller
@RequestMapping("/private/board/list/card")
public class BoardListCardController {
	@Autowired
	private BoardListCardService boardListCardService;
	@Autowired
	private FilesService filesService;
		@Autowired
	private BoardService boardService;
	
	@InitBinder
	protected void initBinder(WebDataBinder binder) {
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        dateFormat.setLenient(false);
        binder.registerCustomEditor(Date.class, new CustomDateEditor(dateFormat, true));
	}
	/**
	 * save
	 * @param board
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/save")
	public  BaseMessage<?> save(BoardListCard listCard,String[] fileId) {
		boardListCardService.save(listCard,fileId);
		return MessageHandler.createSuccessVo("operate successfully");
	}
	
	@RequestMapping("/add")
	public String add(ModelMap model,Integer listId) {
		model.put("listId", listId);
		return "page/team-board-card-add";
	}
	/**
	 * edit
	 * @param model
	 * @param cardId
	 * @return
	 */
	@RequestMapping("/edit")
	public String edit(ModelMap model,Integer cardId) {
		//get board list
		BoardListCard listCard = boardListCardService.getDetailById(cardId);
		model.put("cardId", cardId);
		model.put("data", listCard);
		//list
		BoardList list = boardService.getBoardListDetailById(listCard.getListId());
		model.put("list", list);
		//board
//		Board board = boardService.getDetailByid(list.getBoardId());
//		model.put("board", board);
		//files
		List<Files> filesList = filesService.getListByCard(cardId);
		model.put("filesList", filesList);
		return "page/team-board-card";
	}
}
